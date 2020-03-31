import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    def __init__(self, model_dim, hidden_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(in_features=model_dim, out_features=hidden_dim)
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=model_dim)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, input):
        '''
        Parameters:
        ----------
        input : tensor, shape (batch_size, seq_len, model_dim)
        '''
        out = F.relu(self.linear_1(input))
        out = self.linear_2(out)
        out = self.dropout(out)
        return out

# ===================================== Simple Test =====================================
def test_ffn():
    ffn = PositionwiseFeedForward(model_dim=512, hidden_dim=2048)
    X = torch.randn(100, 20, 512)
    out = ffn(X)
    print(out.shape)

if __name__ == "__main__":
    test_ffn()