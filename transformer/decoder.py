import torch
import torch.nn as nn
import positional_encoding
import decoder_layer

class Decoder(nn.Module):
    def __init__(self, input_dim, model_dim, n_head, key_dim, value_dim, hidden_dim, n_layers, max_seq_len=200, dropout=0.1):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(num_embeddings=input_dim, embedding_dim=model_dim)
        self.pos_enc = positional_encoding.PositionalEncoding(model_dim=model_dim, max_seq_len=200)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            decoder_layer.DecoderLayer(model_dim=model_dim, 
                                        n_head=n_head, 
                                        key_dim=key_dim, 
                                        value_dim=value_dim, 
                                        hidden_dim=hidden_dim)
             for _ in range(n_layers)]
             )

    def forward(self, input, input_mask, encoder_output, encoder_mask):
        '''
        Parameters:
        -----------
        input : tensor, shape (batch_size, seq_len_1)
        input_mask : tensor, shape (batch_size, seq_len_1)
        encoder_output : tensor, shape (batch_size, seq_len_2, model_dim)
        '''
        decoding = self.dropout(self.pos_enc(self.embed(input)))
        self_attn_score_list = []
        enc_attn_score_list = []
        for layer in self.layer_stack:
            decoding, self_attn_scores, enc_attn_scores = layer(decoding, encoder_output, self_attn_mask=input_mask, enc_attn_mask=encoder_mask)
            self_attn_score_list += [self_attn_scores]
            enc_attn_score_list += [enc_attn_scores]
        return decoding, self_attn_score_list, enc_attn_score_list

# ================================= Simple Test ============================================
def test_decoder():
    vocab_size = 100
    import encoder
    encoder = encoder.Encoder(input_dim=vocab_size, model_dim=512, n_head=8, key_dim=64, value_dim=64, hidden_dim=2048, n_layers=6)
    decoder = Decoder(input_dim=vocab_size, model_dim=512, n_head=8, key_dim=64, value_dim=64, hidden_dim=2048, n_layers=6)
    
    src_seq = torch.randint(low=0, high=vocab_size, size=(10, 200))
    trg_seq = torch.randint(low=0, high=vocab_size, size=(10, 20))
    encoder_output, attn_score_list  = encoder(src_seq)
    decoder_output, self_attn_score_list, enc_attn_score_list  = decoder(trg_seq, encoder_output)
    print(
        'decoder output shape:', decoder_output.shape, '\n',
        'len(self attention list):', len(self_attn_score_list), '\n',
        'self attention shape:', self_attn_score_list[0].shape, '\n',
        'len(encoder attention list):', len(enc_attn_score_list), '\n',
        'encoder attention shape:', enc_attn_score_list[0].shape,'\n',
        )

if __name__ == '__main__':
    test_decoder()