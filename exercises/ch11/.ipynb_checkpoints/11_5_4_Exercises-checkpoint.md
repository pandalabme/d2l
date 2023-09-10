# 1. Visualize attention weights of multiple heads in this experiment.


```python
import sys
import torch.nn as nn
import torch
import warnings
from sklearn.model_selection import ParameterGrid
sys.path.append('/home/jovyan/work/d2l_solutions/notebooks/exercises/d2l_utils/')
import d2l
from torchsummary import summary
warnings.filterwarnings("ignore")

class MultiHeadAttention(d2l.Module):  #@save
    """Multi-head attention."""
    def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_k = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_v = nn.LazyLinear(num_hiddens, bias=bias)
        self.W_o = nn.LazyLinear(num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # Shape of queries, keys, or values:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output queries, keys, or values:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # Shape of output: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)
    
    def transpose_qkv(self, X):
        """Transposition for parallel computation of multiple attention heads."""
        # Shape of input X: (batch_size, no. of queries or key-value pairs,
        # num_hiddens). Shape of output X: (batch_size, no. of queries or
        # key-value pairs, num_heads, num_hiddens / num_heads)
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        # Shape of output X: (batch_size, num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        X = X.permute(0, 2, 1, 3)
        # Shape of output: (batch_size * num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        """Reverse the operation of transpose_qkv."""
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)
```

    /home/jovyan/work/d2l_solutions/notebooks/exercises/d2l_utils/d2l.py:131: SyntaxWarning: assertion is always true, perhaps remove parentheses?
      assert(self, 'net'), 'Neural network is defined'
    /home/jovyan/work/d2l_solutions/notebooks/exercises/d2l_utils/d2l.py:135: SyntaxWarning: assertion is always true, perhaps remove parentheses?
      assert(self, 'trainer'), 'trainer is not inited'



```python
num_hiddens, num_heads = 10, 5
attention = MultiHeadAttention(num_hiddens, num_heads, 0.5)
batch_size, num_queries, num_kvpairs = 1, 4, 6
valid_lens = torch.tensor([3,])
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
atten = attention(X, Y, Y, valid_lens)
print(atten.shape)
d2l.show_heatmaps(
    atten.unsqueeze(0).cpu(),
    xlabel='Key positions', ylabel='Query positions')
```

    torch.Size([1, 4, 10])



    
![svg](11_5_4_Exercises_files/11_5_4_Exercises_2_1.svg)
    



```python
print(attention.attention.attention_weights.shape)
d2l.show_heatmaps(
    attention.attention.attention_weights.unsqueeze(0).cpu(),
    xlabel='Key positions', ylabel='Query positions',figsize=(8, 3))
```

    torch.Size([5, 4, 6])



    
![svg](11_5_4_Exercises_files/11_5_4_Exercises_3_1.svg)
    


# 2. Suppose that we have a trained model based on multi-head attention and we want to prune less important attention heads to increase the prediction speed. How can we design experiments to measure the importance of an attention head?

Measuring the importance of attention heads in a multi-head attention model is crucial when considering pruning to increase prediction speed. To design experiments for assessing attention head importance, you can follow these steps:

1. **Attention Head Scores**:
   
   Define a metric or score that quantifies the importance of each attention head. Common metrics include:

   - **Attention Weight Norm**: Calculate the L1/L2 norm (Euclidean norm) of the attention weights for each head. Heads with higher norms might be considered more important.
   
   - **Attention Weight Entropy**: Compute the entropy of the attention weight distribution for each head. Higher entropy suggests more uniform attention distribution, which may indicate less importance.

   - **Attention Weight Variance**: Calculate the variance of the attention weights across different positions and input tokens for each head. High variance may imply a head is more informative.

   - **Output Variance**: Examine the variance of the output representations produced by each head. Higher variance could indicate that the head captures more information.

2. **Create Attention Head Pruning Candidates**:

   Generate several variations of your model, each with a different subset of attention heads pruned (e.g., 50%, 30%, 10% pruning rates). These models should have varying levels of attention head retention.

3. **Evaluate Model Performance**:

   Evaluate the performance of each model on your task of interest (e.g., classification, translation) using a validation or test dataset. Record metrics such as accuracy, BLEU score, or other relevant evaluation criteria.

4. **Importance vs. Performance**:

   Plot or visualize the relationship between attention head importance scores and model performance. You can create scatter plots, bar charts, or other visualizations to observe how changes in attention head importance relate to changes in performance.

5. **Threshold Selection**:

   Based on the importance-performance trade-off, choose a threshold or criteria for pruning. You might select a threshold that maximizes speedup while maintaining a reasonable drop in performance.

6. **Pruning and Speed Evaluation**:

   Implement pruning of less important attention heads according to the selected threshold. Re-evaluate the pruned model's performance and measure prediction speed (inference time) on your validation or test dataset.

7. **Comparative Analysis**:

   Compare the pruned model's performance and prediction speed with the original, unpruned model. Analyze the trade-off between prediction speed and performance to determine the effectiveness of attention head pruning.

8. **Iterative Refinement**:

   If the initial pruning results are not satisfactory, you can iterate by adjusting the importance metric or changing the pruning threshold and evaluating the impact on performance and prediction speed.

9. **Cross-Validation**:

   Perform cross-validation experiments to ensure the robustness of your findings across different data splits.

10. **Reporting and Decision**:

    Report the results of your experiments, including the selected pruning strategy and its impact on prediction speed and model performance. Make a decision on whether the trade-off is acceptable for your specific use case.

By systematically designing experiments to measure the importance of attention heads and their impact on model performance, you can make informed decisions about pruning to increase prediction speed while maintaining reasonable task performance.
