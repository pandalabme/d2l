# 1. Test the fastText results using TokenEmbedding('wiki.en').


```python
import os
import torch
from torch import nn
import sys
sys.path.append('/home/jovyan/work/d2l_solutions/notebooks/exercises/d2l_utils/')
import d2l

class TokenEmbedding:
    """Token Embedding."""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)
    
def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = torch.mv(W, x.reshape(-1,)) / (
        torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
        torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]

def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # Exclude the input word
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
        
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # Remove unknown words
```


```python
d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                           'c1816da3821ae9f43899be655002f6c723e91b88')
ft_wiki = TokenEmbedding('wiki.en')
get_similar_tokens('chip', 3, ft_wiki)
```


```python
get_analogy('man', 'woman', 'son', ft_wiki)
```

# 2. When the vocabulary is extremely large, how can we find similar words or complete a word analogy faster?

When the vocabulary is extremely large, finding similar words or completing a word analogy can be very time-consuming, because we need to compare the word vectors of every word in the vocabulary with the query word or the analogy words. To speed up this process, we can use some techniques such as:

- Indexing: We can use some data structures or algorithms, such as hash tables, trees, or approximate nearest neighbor search, to index the word vectors and reduce the search space. This way, we can find the most similar words or the best analogy words without scanning the whole vocabulary.
- Dimensionality reduction: We can use some methods, such as principal component analysis, singular value decomposition, or autoencoders, to reduce the dimensionality of the word vectors and preserve the most important information. This way, we can reduce the computational cost and memory usage of the similarity or analogy tasks.
- Subword information: We can use some models, such as fastText or BPE, to represent words as sequences of subwords, such as character n-grams, and learn embeddings for each subword. This way, we can handle out-of-vocabulary words and capture the morphological information of words, which can improve the quality and efficiency of the similarity or analogy tasks.
