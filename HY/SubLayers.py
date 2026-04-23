import torch.nn as nn
import torch.nn.functional as F

from Modules import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    # d_model = 256, n_head = 4, d_k = 64, d_v = 64, dropout = 0.1
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q, k, v：enc_input,  # [batch，seq_len，d_model]
        # mask：slf_attn_mask  # [batch,len_s,len_s,2]
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head # 64,64,4
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1) # batch、seq_len、seq_len、seq_len
        residual = q # [batch，seq_len，d_model]
        if self.normalize_before: # 层归一化
            q = self.layer_norm(q)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting. [batch,1,len_s,len_s,2]
        # output： sz_b, n_head，len_v, d_v     attn：sz_b, n_head, len_v, len_v
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1) # (sz_b, len_q, n_head * d_v)
        output = self.dropout(self.fc(output)) # (sz_b, len_q，d_model)
        output += residual # 处理后的 output 张量与原始的 residual 张量相加，实现残差连接。这确保了模型在训练过程中能够保留从输入到输出的直接路径，有助于梯度流动，并且使得网络能够学习到恒等映射。

        if not self.normalize_before:
            output = self.layer_norm(output)
        # output(sz_b, len_q，d_model)   attn(sz_b, n_head, len_v, len_v)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    # d_in：d_model = 256, d_hid：d_inner = 1024
    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (sz_b, len_q，d_model)
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x # (sz_b, len_q，d_model)
