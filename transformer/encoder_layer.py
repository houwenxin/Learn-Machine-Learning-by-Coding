import torch
import torch.nn as nn
import attention
import feed_forward

class EncoderLayer(nn.Module):
    def __init__(self, model_dim, n_head, key_dim, value_dim, hidden_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = attention.MultiHeadAttention(model_dim=model_dim, n_head=n_head, key_dim=key_dim, value_dim=value_dim, dropout=dropout)
        self.ffn = feed_forward.PositionwiseFeedForward(model_dim=model_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=model_dim, eps=1e-12) 
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=model_dim, eps=1e-12)  

    def forward(self, input, mask=None):
        '''
        Parameters:
        -----------
        input : tensor, shape (batch_size, seq_len, model_dim)
        '''
        residual = input
        out, attn_scores = self.self_attention(query=input, key=input, value=input, mask=mask)
        out += residual
        out = self.layer_norm_1(out)

        residual = out
        out = self.ffn(out)
        out += residual
        out = self.layer_norm_2(out)

        return out, attn_scores
    

# ===================================== Simple Test =====================================
def test_enc_layer():
    enc_layer = EncoderLayer(model_dim=512, n_head=8, key_dim=64, value_dim=64, hidden_dim=2048)
    X = torch.randn(100, 20, 512)
    out, attn_scores = enc_layer(X)
    print(out.shape, attn_scores.shape)

if __name__ == "__main__":
    test_enc_layer()