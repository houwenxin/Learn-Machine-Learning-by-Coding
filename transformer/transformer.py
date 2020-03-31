import torch
import torch.nn as nn
import encoder
import decoder

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    batch_size, seq_len = seq.size()
    subsequent_mask = torch.tril(input=torch.ones(size=(1, seq_len, seq_len), device=seq.device)).bool()
    return subsequent_mask
    
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, model_dim, n_head, key_dim, value_dim, hidden_dim, n_layers, pad_idx, max_seq_len=200, dropout=0.1):
        super(Transformer, self).__init__()
        self.pad_idx = pad_idx
        self.encoder = encoder.Encoder(input_dim=input_dim, 
                                        model_dim=model_dim, 
                                        n_head=n_head, 
                                        key_dim=key_dim, 
                                        value_dim=value_dim, 
                                        n_layers=n_layers,
                                        hidden_dim=hidden_dim,
                                        max_seq_len=max_seq_len,
                                        dropout=dropout)
        self.decoder = decoder.Decoder(input_dim=input_dim, 
                                        model_dim=model_dim, 
                                        n_head=n_head, 
                                        key_dim=key_dim, 
                                        value_dim=value_dim, 
                                        n_layers=n_layers,
                                        hidden_dim=hidden_dim,
                                        max_seq_len=max_seq_len,
                                        dropout=dropout)
        self.linear_output = nn.Linear(in_features=model_dim, out_features=output_dim)

    def forward(self, src_seq, trg_seq):
        '''
        Parameters:
        -----------
        src_seq : tensor, shape (batch_size, src_seq_len)
        trg_seq : tensor, shape (batch_size, trg_seq_len)

        Returns:

        '''
        src_mask = get_pad_mask(src_seq, pad_idx=self.pad_idx)
        trg_mask = get_pad_mask(trg_seq, pad_idx=self.pad_idx) & get_subsequent_mask(trg_seq)
        encoder_output, attn_score_list = self.encoder(src_seq, mask=src_mask) # encoding shape (batch_size, src_seq_len, model_dim)
        decoder_output, self_attn_score_list, enc_attn_score_list = self.decoder(input=trg_seq, 
                                                                                input_mask=trg_mask,
                                                                                encoder_output=encoder_output, 
                                                                                encoder_mask=src_mask) 
                                                                                # decoder_output shape (batch_size, trg_seq_len, model_dim)
        out = self.linear_output(decoder_output) # shape (batch_size, trg_seq_len, output_dim)
        return out


# ====================================== Simple Test ================================
def test_transformer():
    vocab_size = 12
    pad_idx = 0
    src_seq = torch.randint(low=0, high=vocab_size, size=(2, 10))
    src_mask = get_pad_mask(src_seq, pad_idx)
    trg_seq = torch.randint(low=0, high=vocab_size, size=(2, 5))
    trg_mask = get_pad_mask(trg_seq, pad_idx) & get_subsequent_mask(trg_seq)

    model = Transformer(input_dim=vocab_size, 
                        output_dim=vocab_size, 
                        model_dim=512, 
                        n_head=8, 
                        key_dim=64, 
                        value_dim=64, 
                        hidden_dim=2048, 
                        n_layers=6,
                        pad_idx=0,
                        )

    pred = model(src_seq=src_seq, trg_seq=trg_seq)
    print(trg_seq.shape, pred.shape)

if __name__ == '__main__':
    test_transformer()