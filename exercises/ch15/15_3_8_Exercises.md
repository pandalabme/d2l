# 1. How does the running time of code in this section changes if not using subsampling?


```python
import time
import collections
import math
import os
import random
import torch
import warnings
import sys
import pandas as pd
sys.path.append('/home/jovyan/work/d2l_solutions/notebooks/exercises/d2l_utils/')
import d2l
from torchsummary import summary
warnings.filterwarnings("ignore")

#@save
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')
#@save
class RandomGenerator:
    """Randomly draw among {1, ..., n} according to n sampling weights."""
    def __init__(self, sampling_weights,k=10000):
        # Exclude
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0
        self.k = k

    def draw(self):
        if self.i == len(self.candidates):
            # Cache `k` random sampling results
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=self.k)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]
    
#@save
def subsample(sentences, vocab,flag=True):
    """Subsample high-frequency words."""
    # Exclude unknown tokens ('<unk>')
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = collections.Counter([
        token for line in sentences for token in line])
    num_tokens = sum(counter.values())

    # Return True if `token` is kept during subsampling
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))
    if flag:
        return ([[token for token in line if keep(token)] for line in sentences],
            counter)
    return (sentences,counter)

#@save
def get_centers_and_contexts(corpus, max_window_size):
    """Return center words and context words in skip-gram."""
    centers, contexts = [], []
    for line in corpus:
        # To form a "center word--context word" pair, each sentence needs to
        # have at least 2 words
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # Context window centered at `i`
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # Exclude the center word from the context words
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts

#@save
def read_ptb():
    """Load the PTB dataset into a list of text lines."""
    data_dir = d2l.download_extract('ptb')
    # Read the training set
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

#@save
def get_negatives(all_contexts, vocab, counter, K, k=10000):
    """Return noise words in negative sampling."""
    # Sampling weights for words with indices 1, 2, ... (index 0 is the
    # excluded unknown token) in the vocabulary
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights,k)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # Noise words cannot be context words
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

#@save
def batchify(data):
    """Return a minibatch of examples for skip-gram with negative sampling."""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (torch.tensor(centers).reshape((-1, 1)), torch.tensor(
        contexts_negatives), torch.tensor(masks), torch.tensor(labels))

#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words, flag=True, k=10000):
    """Download the PTB dataset and then load it into memory."""
    # num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab,flag)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words, k=k)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
                                      collate_fn=batchify)
    return data_iter, vocab
```


```python
t0 = time.time()
data_iter, vocab = load_data_ptb(512, 5, 5)
t1 = time.time()
t1-t0
# names = ['centers', 'contexts_negatives', 'masks', 'labels']
# for batch in data_iter:
#     for name, data in zip(names, batch):
#         print(name, 'shape:', data.shape)
#     break
```




    9.802619218826294




```python
t0 = time.time()
data_iter, vocab = load_data_ptb(512, 5, 5,flag=False)
t1 = time.time()
t1-t0
```




    23.945112943649292



# 2. The RandomGenerator class caches k random sampling results. Set k to other values and see how it affects the data loading speed.


```python
ts = []
k_list = [10,100,1000,10000,100000]
for k in k_list:
    t0 = time.time()
    data_iter, vocab = load_data_ptb(512, 5, 5, k)
    t1 = time.time()
    ts.append(t1-t0)
df = pd.DataFrame({'k':k_list,'time':ts})
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>k</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>10.338631</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100</td>
      <td>9.933641</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1000</td>
      <td>9.871675</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000</td>
      <td>10.212862</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100000</td>
      <td>10.313871</td>
    </tr>
  </tbody>
</table>
</div>



# 3. What other hyperparameters in the code of this section may affect the data loading speed?


```python
ts = []
noise_list = [2,5,10,15,20,25,30]
for num_noise_words in noise_list:
    t0 = time.time()
    data_iter, vocab = load_data_ptb(512, 5, num_noise_words)
    t1 = time.time()
    ts.append(t1-t0)
df = pd.DataFrame({'num_noise_words':noise_list,'time':ts})
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_noise_words</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>6.078225</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>9.767658</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>16.298754</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>22.715422</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>28.570359</td>
    </tr>
    <tr>
      <th>5</th>
      <td>25</td>
      <td>35.331429</td>
    </tr>
    <tr>
      <th>6</th>
      <td>30</td>
      <td>41.231029</td>
    </tr>
  </tbody>
</table>
</div>




```python
ts = []
window_list = [2,5,10,15,20,25,30]
for max_window_size in window_list:
    t0 = time.time()
    data_iter, vocab = load_data_ptb(512, max_window_size, 5)
    t1 = time.time()
    ts.append(t1-t0)
df = pd.DataFrame({'max_window_size':max_window_size,'time':ts})
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>max_window_size</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>7.211373</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>10.097298</td>
    </tr>
    <tr>
      <th>2</th>
      <td>30</td>
      <td>12.955909</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>15.322590</td>
    </tr>
    <tr>
      <th>4</th>
      <td>30</td>
      <td>15.998776</td>
    </tr>
    <tr>
      <th>5</th>
      <td>30</td>
      <td>16.585841</td>
    </tr>
    <tr>
      <th>6</th>
      <td>30</td>
      <td>16.927828</td>
    </tr>
  </tbody>
</table>
</div>


