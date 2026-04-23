import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature # 一个标量，用于缩放点积计算，以控制注意力分数的平滑程度
        self.dropout = nn.Dropout(attn_dropout) # 注意力权重的dropout比率，默认为0.2，用于正则化以防止过拟合

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3)) # (sz_b, n_head, len_v, len_v)
        if mask is not None:
            attn = attn.masked_fill(mask, -1e9) # 如果提供了掩码，将掩码为True的位置的注意力分数设置为一个非常大的负数（如-1e9），这样在应用softmax时，这些位置的权重将接近于0
        attn = self.dropout(F.softmax(attn, dim=-1))
        # sz_b, n_head，len_v, d_v
        output = torch.matmul(attn, v)
        return output, attn
