# 1. Implement distance-based attention by modifying the DotProductAttention code. Note that you only need the squared norms of the keys $\|k_i\|^2$ for an efficient implementation.


```python
import torch.nn as nn
import torch
import math

def masked_softmax(X, valid_lens):  #@save
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)
    
class DistanceAttention(nn.Module):  #@save
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        key_norms = torch.sum(keys ** 2, dim=-1)/ math.sqrt(d) # (batch_size, num_keys)
        scores = scores - 0.5*key_norms.unsqueeze(1) # (batch_size, num_queries, num_keys)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
```


```python
queries = torch.normal(0, 1, (2, 1, 2))
keys = torch.normal(0, 1, (2, 10, 2))
values = torch.normal(0, 1, (2, 10, 4))
valid_lens = torch.tensor([2, 6])

attention = DistanceAttention(dropout=0.5)
attention.eval()
attention(queries, keys, values, valid_lens).shape
```




    torch.Size([2, 1, 4])



# 2. Modify the dot product attention to allow for queries and keys of different dimensionalities by employing a matrix to adjust dimensions.


```python
class DiffDimDotProductAttention(nn.Module):  #@save
    """Scaled dot product attention."""
    def __init__(self, num_hiddens, dropout):
        super().__init__()
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        queries = self.W_q(queries)
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
```


```python
queries = torch.normal(0, 1, (2, 1, 10))
keys = torch.normal(0, 1, (2, 10, 3))
values = torch.normal(0, 1, (2, 10, 4))
valid_lens = torch.tensor([2, 6])

attention = DiffDimDotProductAttention(keys.shape[-1], dropout=0.5)
attention.eval()
attention(queries, keys, values, valid_lens).shape
```




    torch.Size([2, 1, 4])



# 3. How does the computational cost scale with the dimensionality of the keys, queries, values, and their number? What about the memory bandwidth requirements?

The computational cost of self-attention depends on the dimensionality and the number of the keys, queries, and values. Let n be the number of keys, queries, and values, and d be their dimensionality. Then, the computational cost of self-attention is:

- O(n d^2) for computing the query, key, and value matrices by linearly transforming the input matrix.
- O(n^2 d) for computing the dot product between the query and key matrices.
- O(n^2 d) for computing the weighted sum of the value matrix.

Therefore, the total computational cost of self-attention is O(n^2 d + n d^2), which scales quadratically with n and linearly with d.

The memory bandwidth requirements of self-attention are:

- O(n d) for storing the input matrix.
- O(n d) for storing the query, key, and value matrices.
- O(n^2) for storing the attention matrix.
- O(n d) for storing the output matrix.

Therefore, the total memory bandwidth requirements of self-attention are O(n^2 + n d), which scales quadratically with n and linearly with d.

You can find more details about the computational complexity of self-attention in [this paper](https://arxiv.org/pdf/1706.03762v5.pdf).

- (1) Attention Is All You Need - arXiv.org. https://arxiv.org/pdf/1706.03762v5.pdf.
- (2) Computational Complexity of Self-Attention in the .... https://stackoverflow.com/questions/65703260/computational-complexity-of-self-attention-in-the-transformer-model.
- (3) Dynamic Convolution: Attention over Convolution Kernels .... https://arxiv.org/pdf/1912.03458.pdf.
