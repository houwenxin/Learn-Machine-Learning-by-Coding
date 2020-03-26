import torch.nn as nn

class PositionalEncoding(nn.Module):
    '''
    Positional Encoding:
    PE(pos, 2i) = sin(pos/10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))

    Parameters :
    ------------
    enc_dim : int, encoding dim.
    max_seq_len : int, max length of input sequences.

    Attributes:
    ------------
    pe_table : torch.Tensor, shape (), Table of positional encodings.
    '''
    def __init__(self, model_dim, max_seq_len=200):
        super(PositionalEncoding).__init__()
        
    def forward(self, input):
        '''
        Parameters: 
        -----------
        input : torch.Tensor, shape (batch_size, seq_len, model_dim), embeddings of input sequences.
        '''
        pass

class Encoder(nn.Module):
    '''
    Transformer Encoder.

    Paramters:
    ----------
    input_dim : int, dimension of input features.
    model_dim : int, dimension of attention embeddings.
    '''
    def __init__(self, input_dim, model_dim=512):
        super(Encoder).__init__()
        self.embed = nn.Embedding(num_embeddings=input_dim, embedding_dim=model_dim)
        self.pos_enc = PositionalEncoding(model_dim=model_dim)

    def forward(self, input):
        '''
        Parameters:
        ---------
        input : torch.Tensor, shape (batch_size, seq_len)

        Returns:
        
        '''
        pass
    