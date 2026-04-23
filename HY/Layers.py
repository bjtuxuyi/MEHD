import torch.nn as nn
from SubLayers import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """ Compose with two layers """
    # d_model=64, d_rnn=256, d_inner=128,n_layers=4, n_head=4, d_k=16, d_v=16, dropout=0.1,device=None,loc_dim=2,CosSin=True
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        # enc_input,  # batch*mslb*d_model
        # non_pad_mask  # batch, mslb, 1
        # slf_attn_mask  # [batch,mslb,mslb,2]
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)  # enc_output(batch, mslb，d_model)   enc_slf_attn(batch, n_head, mslb, mslb)

        enc_output *= non_pad_mask # *元素级乘法 (batch, mslb，d_model)*（batch, mslb, 1） --> (batch, mslb，d_model)
        enc_output = self.pos_ffn(enc_output) # (batch, mslb，d_model)
        enc_output *= non_pad_mask # (batch, mslb，d_model)

        return enc_output, enc_slf_attn # # (sz_b, len_q，d_model)    (batch, n_head, mslb, mslb)

class EncoderLayer2(nn.Module):
    # d_model=64, d_rnn=256, d_inner=128,n_layers=4, n_head=4, d_k=16, d_v=16, dropout=0.1,device=None,loc_dim=2,CosSin=True
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super(EncoderLayer2, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input_Q,enc_input_K_V, main_non_pad_mask=None, cross_attn_mask=None):
        # enc_input,  # batch*mslb*d_model
        # non_pad_mask  # batch, mslb, 1
        # slf_attn_mask  # [batch,mslb,mslb,2]
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input_Q, enc_input_K_V, enc_input_K_V, mask=cross_attn_mask)  # enc_output(batch, mslb，d_model)   enc_slf_attn(batch, n_head, mslb, mslb)

        enc_output *= main_non_pad_mask # *元素级乘法 (batch, mslb，d_model)*（batch, mslb, 1） --> (batch, mslb，d_model)
        enc_output = self.pos_ffn(enc_output) # (batch, mslb，d_model)
        enc_output *= main_non_pad_mask # (batch, mslb，d_model)

        return enc_output, enc_slf_attn # # (sz_b, len_q，d_model)    (batch, n_head, mslb, mslb)


