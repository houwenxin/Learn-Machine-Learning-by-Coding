import torch
import torch.nn as nn
import positional_encoding
import encoder_layer

class Encoder(nn.Module):
    '''
    Transformer Encoder.

    Data Flow: 
    one hot vectors
    --> emedding + position encoding --> dropout
    --> n_layers * (multi-head attention --> dropout --> add & norm --> positionwise feed forward --> dropout --> add & norm)
    --> encoding

    Paramters:
    ----------
    input_dim : int, dimension of input features.
    model_dim : int, dimension of attention embeddings.
    '''
    def __init__(self, input_dim, model_dim, n_head, key_dim, value_dim, hidden_dim, n_layers, max_seq_len=200, dropout=0.1):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(num_embeddings=input_dim, embedding_dim=model_dim)
        self.pos_enc = positional_encoding.PositionalEncoding(model_dim=model_dim, max_seq_len=max_seq_len)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            encoder_layer.EncoderLayer(model_dim=model_dim, n_head=n_head, key_dim=key_dim, value_dim=value_dim, hidden_dim=hidden_dim)
            for _ in range(n_layers)])

    def forward(self, input, mask):
        '''
        Parameters:
        -----------
        input : torch.Tensor, shape (batch_size, seq_len)

        Returns:
        --------
        encoding : tensor, shape (batch_size, seq_len, model_dim), encoding output of transformer encoder.
        attn_score_list : list of tensors, list of attention scores [shape (batch_size, seq_len, seq_len)] in multi-head attention layers. 
        '''
        encoding = self.dropout(self.pos_enc(self.embed(input)))
        attn_score_list = []
        for layer in self.layer_stack:
            encoding, attn_scores = layer(encoding, mask=mask)
            attn_score_list += [attn_scores]
        return encoding, attn_score_list


# ===================================== Simple Test =====================================
def test_encoder():
    vocab_size = 10
    encoder = Encoder(input_dim=vocab_size, model_dim=512, n_head=8, key_dim=64, value_dim=64, hidden_dim=2048, n_layers=6)
    seq = torch.randint(low=0, high=vocab_size, size=(10, 200))
    mask = (seq != 0).unsqueeze(-2) # shape (batch_size, 1, seq_len)
    encoding, attn_score_list  = encoder(seq, mask=mask)
    print(encoding.shape, 'len(attention list):', len(attn_score_list), attn_score_list[0].shape)

if __name__ == '__main__':
    test_encoder()