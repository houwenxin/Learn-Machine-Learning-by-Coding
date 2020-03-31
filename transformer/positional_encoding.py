import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    '''
    Positional Encoding (i is the dimension, pos is the postion):
    PE(pos, 2i) = sin(pos/10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))

    Parameters :
    ------------
    model_dim : int, encoding dim.
    max_seq_len : int, max length of input sequences.

    Attributes:
    ------------
    pe_table : tensor, shape (max_seq_len, enc_dim), Table of positional encodings.
    '''
    def __init__(self, model_dim, max_seq_len=200):
        super(PositionalEncoding, self).__init__()
        #self.pe_table = self._get_pe_table(model_dim, max_seq_len)
        self.register_buffer('pe_table', self._get_pe_table(model_dim, max_seq_len))

    def forward(self, input):
        '''
        Parameters: 
        -----------
        input : tensor, shape (batch_size, seq_len, model_dim), embeddings of input sequences.
        '''
        return input + self.pe_table[:input.shape[1]].clone().detach()

    def _get_pe_table(self, model_dim, max_seq_len):
        '''
        div_term : 1 / 10000^(2i/d_model) = exp(-log(10000) * 2i / d_model)

        Returns:
        ---------
        pe_table : tensor, shape (max_seq_len, enc_dim), Table of positional encodings.
        '''
        pe_table = torch.zeros((max_seq_len, model_dim))
        position = torch.arange(0, max_seq_len, dtype=torch.float32) # shape (max_seq_len,)
        div_term = torch.exp(-torch.log(torch.tensor(10000.)) * torch.arange(0, model_dim, 2, dtype=torch.float32) / model_dim) # shape (model_dim/2, )
        
        # Align shapes
        position = position.unsqueeze(1) # (max_seq_len, 1)
        div_term = div_term.unsqueeze(0) # (1, model_dim/2)
        pe_table[:, 0::2] = torch.sin(position * div_term)
        pe_table[:, 1::2] = torch.cos(position * div_term)
        return pe_table



# ===================================== Simple Test =====================================
def test_pe():
    import matplotlib.pyplot as plt
    pe = PositionalEncoding(model_dim=512, max_seq_len=256)
    X = torch.zeros(14, 15, 512)
    y = pe(X)

    plt.plot(y[0, :, 0])
    plt.plot(y[0, :, 1])
    plt.show()

if __name__ == '__main__':
    test_pe()