import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, key_dim, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.key_dim = key_dim
        self.dropout = nn.Dropout(p=attn_dropout)

    def forward(self, query, key, value, mask=None):
        '''
        Parameters:
        -----------
        query : tensor, shape (batch_size, n_head, seq_len_1, key_dim)
        key : tensor, shape (batch_size, n_head, seq_len_2, key_dim)
        value : tensor, shape (batch_size, n_head, seq_len_2, value_dim)
        mask : tensor, shape (batch_size, seq_len_1, seq_len_2)

        Returns:
        ---------
        out : tensor, shape (batch_size, n_head, seq_len_1, value_dim)
        attn_scores : tensor, shape (batch_size, n_head, seq_len_1, seq_len_2)
        '''
        attn_scores = torch.matmul(query / self.key_dim, key.transpose(-2, -1)) # shape (batch_size, n_head, seq_len_1, seq_len_2)

        if mask is not None:
            mask = mask.unsqueeze(1).eq(0) # mask shape (batch_size, 1, 1, seq_len_2 == key/value len)
            attn_scores = attn_scores.masked_fill(mask=mask, value=-1e9) # set to -inf
            attn_scores = F.softmax(attn_scores, dim=-1).masked_fill(mask=mask, value=0.0)
        else:
            attn_scores = F.softmax(attn_scores, dim=-1)

        attn_scores = self.dropout(attn_scores)
        out = torch.matmul(attn_scores, value) # shape (batch_size, n_head, seq_len_1, value_dim)
        return out, attn_scores

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, n_head, key_dim, value_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.linear_Q = nn.Linear(in_features=model_dim, out_features=key_dim * n_head)
        self.linear_K = nn.Linear(in_features=model_dim, out_features=key_dim * n_head)
        self.linear_V = nn.Linear(in_features=model_dim, out_features=value_dim * n_head)
        self.attention = ScaledDotProductAttention(key_dim=key_dim)
        self.linear_out = nn.Linear(in_features=value_dim * n_head, out_features=model_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        '''
        Parameters:
        -----------
        query : tensor, shape (batch_size, seq_len_1, model_dim)
        key : tensor, shape (batch_size, seq_len_2, model_dim)
        value : tensor, shape (batch_size, seq_len_2, model_dim)

        Returns:
        ---------
        out : tensor, shape (batch_size, seq_len_1, model_dim)
        attn_scores : tensor, shape (batch_size, n_head, seq_len_1, seq_len_2)
        '''
        n_head = self.n_head
        key_dim = self.key_dim
        value_dim = self.value_dim
        batch_size, seq_len_1, model_dim = query.shape
        seq_len_2 = key.shape[1]
        query = self.linear_Q(query).view(batch_size, n_head, seq_len_1, key_dim)
        key = self.linear_K(key).view(batch_size, n_head, seq_len_2, key_dim)
        value = self.linear_V(value).view(batch_size, n_head, seq_len_2, value_dim)

        out, attn_scores = self.attention(query, key, value, mask)
        out = out.view(batch_size, seq_len_1, -1)
        out = self.dropout(self.linear_out(out))
        return out, attn_scores


# ===================================== Simple Test =====================================
def test_attn():
    attention = MultiHeadAttention(model_dim=512, n_head=8, key_dim=64, value_dim=64)
    X = torch.randn(100, 20, 512)
    out, attn_scores = attention(X, X, X)
    print(out.shape, attn_scores.shape)

if __name__ == "__main__":
    test_attn()