import torch
import torch.nn as nn
import attention
import feed_forward

class DecoderLayer(nn.Module):
    def __init__(self, model_dim, n_head, key_dim, value_dim, hidden_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = attention.MultiHeadAttention(model_dim=model_dim, n_head=n_head, key_dim=key_dim, value_dim=value_dim, dropout=dropout)
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=model_dim, eps=1e-12)
        self.encoder_attention = attention.MultiHeadAttention(model_dim=model_dim, n_head=n_head, key_dim=key_dim, value_dim=value_dim, dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=model_dim, eps=1e-12)
        self.ffn = feed_forward.PositionwiseFeedForward(model_dim=model_dim, hidden_dim=hidden_dim)
        self.layer_norm_3 = nn.LayerNorm(normalized_shape=model_dim, eps=1e-12)
        
    def forward(self, input, encoder_output, self_attn_mask=None, enc_attn_mask=None):
        '''
        Parameters:
        -----------
        input : tensor, shape (batch_size, seq_len_1, model_dim)
        encoder_output : tensor, shape (batch_size, seq_len_2, model_dim)

        Returns:
        ----------
        out : tensor, shape (batch_size, seq_len_1, model_dim)
        self_attn_scores : tensor, shape (batch_size, seq_len_1, seq_len_1)
        dec_enc_attn_scores : tensor, shape (batch_size, seq_len_1, seq_len_2)
        '''
        residual = input
        out, self_attn_scores = self.self_attention(query=input, key=input, value=input, mask=self_attn_mask)
        out += residual
        out = self.layer_norm_1(out)

        residual = out
        out, enc_attn_scores = self.encoder_attention(query=out, key=encoder_output, value=encoder_output, mask=enc_attn_mask)
        out += residual
        out = self.layer_norm_2(out)

        residual = out
        out = self.ffn(out)
        out = self.layer_norm_3(out)

        return out, self_attn_scores, enc_attn_scores