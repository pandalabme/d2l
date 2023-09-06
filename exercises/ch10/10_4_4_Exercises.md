# 1. If the different directions use a different number of hidden units, how will the shape of $H_t$ change?

$H_t.shape[-1] = \overrightarrow{H_t}.shape[-1] + \overleftarrow{H_t}.shape[-1]$


```python
import sys
import torch.nn as nn
import torch
import warnings
sys.path.append('/home/jovyan/work/d2l_solutions/notebooks/exercises/d2l_utils/')
import d2l
from torchsummary import summary
warnings.filterwarnings("ignore")
from sklearn.model_selection import ParameterGrid

class BiRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.f_rnn = d2l.RNNScratch(num_inputs, num_hiddens[0], sigma)
        self.b_rnn = d2l.RNNScratch(num_inputs, num_hiddens[1], sigma)
        self.num_hiddens = sum(num_hiddens)  # The output dimension will be doubled
        
    def forward(self, inputs, Hs=None):
        f_H, b_H = Hs if Hs is not None else (None, None)
        f_outputs, f_H = self.f_rnn(inputs, f_H)
        b_outputs, b_H = self.b_rnn(reversed(inputs), b_H)
        outputs = [torch.cat((f, b), -1) for f, b in zip(
            f_outputs, reversed(b_outputs))]
        return outputs, (f_H, b_H)
```

    /home/jovyan/work/d2l_solutions/notebooks/exercises/d2l_utils/d2l.py:129: SyntaxWarning: assertion is always true, perhaps remove parentheses?
      assert(self, 'net'), 'Neural network is defined'
    /home/jovyan/work/d2l_solutions/notebooks/exercises/d2l_utils/d2l.py:133: SyntaxWarning: assertion is always true, perhaps remove parentheses?
      assert(self, 'trainer'), 'trainer is not inited'



```python
bi_rnn = BiRNNScratch(num_inputs=4, num_hiddens=[8,16])
X = torch.randn(1,4)
outputs, (f_H, b_H) = bi_rnn(X)
outputs[0].shape
```




    torch.Size([4, 24])



# 2. Design a bidirectional RNN with multiple hidden layers.


```python
class BiRNN(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens, num_layers):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.RNN(num_inputs, num_hiddens, num_layers=num_layers, bidirectional=True)
        self.num_hiddens *= 2
```


```python
bi_rnn = BiRNN(num_inputs=4, num_hiddens=8,num_layers=2)
bi_rnn
```




    BiRNN(
      (rnn): RNN(4, 8, num_layers=2, bidirectional=True)
    )



# 3. Polysemy is common in natural languages. For example, the word “bank” has different meanings in contexts “i went to the bank to deposit cash” and “i went to the bank to sit down”. How can we design a neural network model such that given a context sequence and a word, a vector representation of the word in the correct context will be returned? What type of neural architectures is preferred for handling polysemy?

One possible way to design a neural network model for word sense disambiguation is to use **contextualized word embeddings** (CWEs). CWEs are vector representations of words that are sensitive to the surrounding context, meaning that the same word can have different embeddings depending on how it is used in a sentence. For example, the word "bank" would have different embeddings in the contexts "I went to the bank to deposit cash" and "I went to the bank to sit down". CWEs can be obtained by using neural language models that are trained on large corpora of text, such as BERT, ELMo, or XLNet¹.

To use CWEs for word sense disambiguation, we can follow a simple but effective approach based on nearest neighbor classification². The idea is to pre-compute the CWEs for each word sense in a given lexical resource, such as WordNet, using gloss definitions and example sentences. Then, given a context sequence and a word, we can compute the CWE for the word in that context using the neural language model. Finally, we can compare the CWE of the word with the pre-computed CWEs of its possible senses and select the nearest one as the predicted sense. This method can achieve state-of-the-art results on several WSD benchmarks².

The type of neural architectures that is preferred for handling polysemy is usually based on **recurrent neural networks** (RNNs), especially those with **long short-term memory** (LSTM) cells. RNNs are able to capture the sequential nature of language and model long-distance dependencies between words. LSTM cells are a special type of RNN units that can learn to store and forget information over time, which is useful for dealing with complex and ambiguous contexts. RNNs with LSTM cells can be used as the backbone of neural language models that produce CWEs, such as ELMo³ or XLNet¹.

- (1) [2106.07967] Incorporating Word Sense Disambiguation in .... https://arxiv.org/abs/2106.07967.
- (2) Neural Network Models for Word Sense Disambiguation: An .... https://www.researchgate.net/publication/324014399_Neural_Network_Models_for_Word_Sense_Disambiguation_An_Overview/fulltext/5b3c024b4585150d23f66a4a/Neural-Network-Models-for-Word-Sense-Disambiguation-An-Overview.pdf.
- (3) arXiv:1909.10430v2 [cs.CL] 1 Oct 2019. https://arxiv.org/pdf/1909.10430.pdf.
- (4) undefined. https://doi.org/10.48550/arXiv.2106.07967.
