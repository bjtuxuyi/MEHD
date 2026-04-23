import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import  Constants
from Layers import EncoderLayer,EncoderLayer2
from SDGCN import *
from SDHGCN import *

def get_non_pad_mask(seq):
    # seq: batch * seq_len

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    # seq_k、seq_q: batch*seq_len*2
    '''为注意力机制生成一个掩码矩阵，用于屏蔽键序列（key sequence）中的填充部分。在处理序列数据时，填充（padding）通常用于确保所有序列具有相同的长度，但模型需要能够区分实际数据和填充的部分。'''
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1, -1)  # b x lq x lk
    return padding_mask # batch*len_q*len_k*2


def get_subsequent_mask(seq, dim=2):
    """创建一个用于屏蔽未来信息的掩码矩阵，常用于处理序列数据，特别是在需要避免在序列解码过程中使用未来信息的场景"""
    # event_loc: batch*seq_len*2
    sz_b, len_s = seq.size()[:2]
    # 这个变量用于创建一个上三角矩阵，其对角线为1，其余为0。在自然语言处理中，这种掩码用于屏蔽未来的位置，以确保在序列生成或编码时，每个元素只能看到它之前的位置。
    subsequent_mask = torch.triu(
        torch.ones((dim, len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1).permute(1,2,0)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1,-1)  # b x ls x ls
    return subsequent_mask # [batch,dim,len_s,len_s]



class SubEventConvModel(nn.Module):
    def __init__(self, d_model, kernel_size=3, hidden_dim=128, dropout=0.1):
        super(SubEventConvModel, self).__init__()
        # 定义一维卷积层
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=hidden_dim, kernel_size=kernel_size, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        # 定义第二层一维卷积层
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=1, kernel_size=1, stride=1)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, sub_event_input):
        # sub_event_input: 次列车事件输入张量，形状为 [batch, mslb-1, 4, d_model]
        batch_size, seq_len, num_trains, d_model = sub_event_input.size()
        # 将输入张量的形状调整为 [batch * mslb-1, 4, d_model]
        sub_event_input = sub_event_input.view(-1, num_trains, d_model)
        # 应用第一层一维卷积
        conv_output = self.conv1(sub_event_input)
        conv_output = self.relu1(conv_output)
        conv_output = self.dropout1(conv_output)
        # 应用第二层一维卷积
        conv_output = self.conv2(conv_output)
        conv_output = self.relu2(conv_output)
        conv_output = self.dropout2(conv_output)
        # 将输出张量的形状从 [batch * mslb-1, 1, d_model] 转换回 [batch, mslb-1, 1, d_model]
        conv_output = conv_output.view(batch_size, seq_len, 1, d_model)

        return conv_output.squeeze(2)

class StrFeature_Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        # print("self.embedding:",self.embedding.weight.shape)
    def forward(self, x, feature_idx):
        if len(x.shape) == 3: # 处理的是每个事件序列
            embedding_x = x[:, :, feature_idx].long()
            total_features = x.size(-1)
            no_embedding_x = x[:, :, sorted(set(range(total_features)) - set(feature_idx))]
            # print("embedding_x:",embedding_x.shape)
            e_feature = self.embedding(embedding_x)
            e_feature = e_feature.reshape(e_feature.shape[0], e_feature.shape[1], -1)
            output = torch.cat([e_feature, no_embedding_x], dim=-1)
            return output
        if len(x.shape) == 2: # 处理的是全部序列
            embedding_x = x[:, feature_idx].long()
            total_features = x.size(-1)
            no_embedding_x = x[:, sorted(set(range(total_features)) - set(feature_idx))]
            e_feature = self.embedding(embedding_x)
            e_feature = e_feature.reshape(e_feature.shape[0], -1)
            output = torch.cat([e_feature, no_embedding_x], dim=-1)
            return output
        else:
            return None


class WeightedFusion(nn.Module):
    def __init__(self, feat_dim=64*3):
        super().__init__()
        self.w1 = nn.Linear(feat_dim, feat_dim, bias=False)  # 线性变换矩阵W1 192*192
        self.w2 = nn.Linear(feat_dim//3, feat_dim, bias=False)  # 线性变换矩阵W2 64*192

    def forward(self, x1, x2):
        return self.w1(x1) + self.w2(x2)


class Encoder_ST(nn.Module):

    def __init__(
            self, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout,device, loc_dim,CosSin = False):
        super().__init__()

        self.device = device
        self.d_model = d_model
        self.loc_dim = loc_dim
        self.operation_embedding_dim = Constants.operation_embedding_dim
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=device)

        self.event_emb_loc = nn.Sequential(
          nn.Linear(self.loc_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
        )
        self.operation_emb = nn.Sequential(
            nn.Linear(self.operation_embedding_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        # 主事件自注意力
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])
        # 空间自注意力
        self.layer_stack_loc = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])
        # 时间自注意力
        self.layer_stack_temporal = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        self.position_vec = self.position_vec.to(time)
        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def generate_exponential_decay(self,event_time,i_o_num):
        s_0, s_1, s_2, s_3 = i_o_num.shape
        decay_rate = 0.2
        result2 = torch.zeros([s_0, s_1, s_2], device=i_o_num.device)
        for batch in range(s_0):
            for i in range(s_1):
                tensor1 = event_time[batch][i + 1]
                tensor2 = i_o_num[batch, i, :, 0]
                result = torch.tensor(
                    [torch.exp(-decay_rate * (tensor1.item() - x)) if x != 0 else tensor1.item() - tensor1.item() for x
                     in tensor2])
                result2[batch, i, :] = result
        result2 = result2.view(s_0, s_1, s_2, 1)
        final = torch.cat((i_o_num, result2), dim=3)
        return final

    def forward(self, event_time, event_loc,event_operation_feature,non_pad_mask,strembedding):
        '''1. 生成注意力掩码矩阵：slf_attn_mask，用于屏蔽未来和填充部分'''
        slf_attn_mask_subseq = get_subsequent_mask(event_loc, dim=self.loc_dim) # 创建一个用于屏蔽未来信息的掩码矩阵，常用于处理序列数据，特别是在需要避免在序列解码过程中使用未来信息的场景
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_loc, seq_q=event_loc) # 为注意力机制生成一个掩码矩阵，用于屏蔽键序列（key sequence）中的填充部分
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0) # .gt(0) 是一个比较操作，它会将所有非零（True，即1）的值转换为True，将零（False，即0）保持为False
        slf_attn_mask = slf_attn_mask[:, :, :, 0]  # 注意力掩码
        '''2.事件编码:时间编码：位置向量；  空间编码：全连接层；  特征编码：映射，全连接'''
        enc_output_temporal = self.temporal_enc(event_time, non_pad_mask)
        enc_output_loc = self.event_emb_loc(event_loc)
        enc_output_features = strembedding(event_operation_feature,[0,1])
        enc_output_operation = self.operation_emb(enc_output_features)
        enc_output = enc_output_temporal+enc_output_loc+enc_output_operation # batch*mslb*d_model
        '''3.注意力计算'''
        for index in range(len(self.layer_stack)):
            enc_output_loc, _ = self.layer_stack_loc[index](enc_output_loc,non_pad_mask=non_pad_mask,slf_attn_mask=slf_attn_mask)
            enc_output_temporal, _ = self.layer_stack_temporal[index](enc_output_temporal,non_pad_mask=non_pad_mask,slf_attn_mask=slf_attn_mask)
            enc_output, _ = self.layer_stack[index](enc_output,non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)
        return enc_output, enc_output_temporal, enc_output_loc

class RNN_layers(nn.Module):
    def __init__(self, d_model, d_rnn, out_dim):
        super().__init__()
        self.d_model = d_model
        self.out_dim = out_dim
        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, self.out_dim)
    def forward(self, data, non_pad_mask):
        # data(sz_b, len_q，d_model)  non_pad_mask：[batch, seq_len, 1]
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        # 如果 batch_first=False（默认），形状为 (seq_len, batch, feature)；如果 batch_first=True，形状为 (batch, seq_len, feature)。
        # enforce_sorted：一个布尔值，指示输入的 lengths 是否已经按降序排序。如果为 True，则假设序列已经是按长度降序排列的，这可以提高效率。
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0] # 解包，由输出序列和输出隐藏状态两部分组成
        out = self.projection(out)
        # (sz_b, len_q，d_model)
        return out

class Transformer_ST(nn.Module):
    def __init__(
            self,adj, d_model=64, d_rnn=256, d_inner=128,
            n_layers=4, n_head=4, d_k=16, d_v=16, dropout=0.1,device="cuda:0",loc_dim=2,CosSin=True):
        super().__init__()
        self.encoder = Encoder_ST(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device=device,
            loc_dim = loc_dim,
            CosSin = CosSin
        )
        self.adj = adj
        self.Hy_conv = SDGCN(
            in_dim=Constants.hy_embedding_dim,
            hidden_dim=d_model,
            out_dim=d_model,
            edge_index=dense_to_coo(self.adj),
            num_nodes=self.adj.shape[0],
            device=device
        )

        # self.Hy_conv2 = SDHGCN(
        #     in_features=Constants.hy_embedding_dim,
        #     out_features=d_model
        # )

        self.external_embedding_dim = Constants.external_embedding_dim
        self.alpha = nn.Parameter(torch.tensor(-0.1))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.rnn = RNN_layers(2*d_model, d_rnn,d_model)
        self.rnn_temporal = RNN_layers(d_model, d_rnn,d_model)
        self.rnn_spatial = RNN_layers(d_model, d_rnn,d_model)

        self.external_emb = nn.Sequential(
            nn.Linear(self.external_embedding_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )


    def forward(self, batch,event_time, event_loc,event_operation_feature, event_external_feature,strembedding,device):
        '''0.掩码分析：获取真正运行的列车事件的表征'''
        non_pad_mask = get_non_pad_mask(event_time)
        lengths = non_pad_mask.squeeze(2).long().sum(1)
        really_operation_trains = (lengths != 0).nonzero().squeeze().tolist()
        really_operation_trains = torch.tensor(really_operation_trains).to(device)
        '''1.基础超边事件序列encode,结合运行数据'''
        B_enc_output, B_enc_output_temporal, B_enc_output_loc = self.encoder(event_time, event_loc,event_operation_feature,non_pad_mask,strembedding) # [69, 14, 64]
        assert (B_enc_output != B_enc_output_temporal).any() & (B_enc_output != B_enc_output_loc).any() & (B_enc_output_loc != B_enc_output_temporal).any()
        '''2.相关超边有向图卷积'''
        event_x = batch.squeeze()[:,[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]].to(device) # 节点特征
        event_x = strembedding(event_x, [3, 4]).to(device)
        R_enc_output = self.Hy_conv(event_x) # ([637, 64])


        '''3.外部因素编码'''
        E_output_features = strembedding(event_external_feature, [2, 3])
        E_enc_output = self.external_emb(E_output_features) #[69, 14, 64]
        '''4.融合数据'''
        # 使用concat的形式连接环境因素和基础事件超边表征
        enc_output = torch.cat((B_enc_output, E_enc_output),dim=-1)
        # 提取really_operation_train的到站数据 即剔除没有运行的列车事件
        enc_output = enc_output[really_operation_trains,:,:]
        enc_output_temporal = B_enc_output_temporal[really_operation_trains,:,:]
        enc_output_loc = B_enc_output_loc[really_operation_trains,:,:]
        really_non_pad_mask = non_pad_mask[really_operation_trains,:,:]
        '''5.Event_LSTM处理有向超边'''
        enc_output = self.rnn(enc_output, really_non_pad_mask) # # (sz_b, len_q，d_model)
        enc_output_temporal = self.rnn_temporal(enc_output_temporal, really_non_pad_mask)
        enc_output_loc = self.rnn_spatial(enc_output_loc, really_non_pad_mask)
        '''6.输出为基础事件序列输出、相关事件序列表征、掩码'''
        enc_output_all = torch.cat((enc_output_temporal, enc_output_loc, enc_output),dim=-1)
        return enc_output_all, R_enc_output, really_non_pad_mask,lengths # (batch, mslb，d_model*3),(event_num,d_model) ，（batch, mslb, 1）

























