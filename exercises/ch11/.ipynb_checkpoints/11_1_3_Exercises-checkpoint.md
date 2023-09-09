```python
!jupyter nbconvert --to markdown 10_7_10_Exercises.ipynb
```


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
```

# 1. Suppose that you wanted to reimplement approximate (key, query) matches as used in classical databases, which attention function would you pick?

One possible attention function that can be used to implement approximate (key, query) matches is the **scaled dot-product attention**¹. This function computes the similarity between the query and each key by taking the dot product and scaling it by the square root of the key dimension. Then, it applies a softmax function to obtain the attention weights, which are used to compute a weighted sum of the values. This function can handle queries and keys of different lengths, and it can learn to attend to the most relevant parts of the keys for each query. 

The scaled dot-product attention function can be defined as follows:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

where $Q$ is a matrix of queries, $K$ is a matrix of keys, $V$ is a matrix of values, and $d_k$ is the dimension of the keys.

- (1) What exactly are keys, queries, and values in attention .... https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms.
- (2) 11.1. Queries, Keys, and Values — Dive into Deep Learning 1 .... https://d2l.ai/chapter_attention-mechanisms-and-transformers/queries-keys-values.html.
- (3) 注意力,多头注意力,自注意力及Pytorch实现 - 知乎. https://zhuanlan.zhihu.com/p/366592542.
- (4) Alignment Attention by Matching Key and Query Distributions. https://arxiv.org/abs/2110.12567.

# 2. Suppose that the attention function is given by $\alpha(q,k_i)=q^Tk_i$ and that $k_i=v_i$ for $i=1,\dots m,$. Denote by $p(k_i;q)$ the probability distribution over keys when using the softmax normalization in (11.1.3). Prove that $\nabla_qAttention(q,D)=Cov_{p(k_i;q)}[k_i]$.

First, let us recall the definition of the attention function and the softmax normalization from (11.1.3):

$$
\text{Attention}(q,D) = \sum_{i=1}^m v_ip(k_i;q)
$$

$$
p(k_i;q) = \frac{\exp(\alpha(q,k_i))}{\sum_{j=1}^m \exp(\alpha(q,k_j))}
$$

where $q$ is the query vector, $D = \{(k_1,v_1),\dots,(k_m,v_m)\}$ is the set of key-value pairs, and $\alpha(q,k_i)$ is the similarity score between $q$ and $k_i$. we assume that $\alpha(q,k_i)=q^Tk_i$
$$
\nabla_qAttention(q,D) = \nabla_q\sum_{i=1}^m v_ip(k_i;q) = \sum_{i=1}^m \nabla_qp(k_i;q)k_i^T
=\sum_{i=1}^m \nabla_q\alpha(q,k_i)p(k_i;q)(1-p(k_i;q))k_i^T=\sum_{i=1}^mk_ip(k_i;q)(1-p(k_i;q))k_i^T$$
The covariance of $k_i$ under the probability distribution $p(k_i;q)$ is:

$$
Cov_{p(k_i;q)}[k_i] = E_{p(k_i;q)}(k_i - E_{p(k_i;q)}[k_i])^2=E_{p(k_i;q)}[(k_i - E_{p(k_i;q)}[k_i])(k_i - E_{p(k_i;q)}[k_i])^T]=E[(k_i-\sum k_ip_i)(k_i-\sum k_ip_i)^T]=E(k_i\sum p_i^Tk_i^T - k_ik_i^T-\sum k_ip_i\sum p_i^Tk_i^T+\sum k_ip_i*k_i^T)=\sum k_ip_i\sum p_i^Tk_i^T - \sum p_ik_ik_i^T-\sum k_ip_i\sum p_i^Tk_i^T+\sum k_ip_i\sum p_i^Tk_i^T=\sum_{i=1}^mk_ip(k_i;q)(1-p(k_i;q))k_i^T=\nabla_qAttention(q,D)
$$

# 3. Design a differentiable search engine using the attention mechanism.

A possible way to design a differentiable search engine using the attention mechanism is as follows:

- First, we need to define a database of documents that can be searched by the engine. For simplicity, we can assume that each document is represented by a vector of features, such as word embeddings, TF-IDF scores, or topic models. We can also assume that the database is fixed and does not change over time.
- Second, we need to define a query model that can generate a query vector from a natural language input. For example, we can use a recurrent neural network (RNN) or a transformer to encode the input into a fixed-length vector. Alternatively, we can use a keyword-based approach to extract the most relevant terms from the input and represent them as vectors.
- Third, we need to define an attention function that can compute the similarity between the query vector and each document vector in the database. For example, we can use the scaled dot-product attention function¹ or the additive attention function². The attention function should output a vector of attention weights, where each weight corresponds to the relevance of a document to the query.
- Fourth, we need to define an output model that can generate a ranked list of documents from the attention weights. For example, we can use a softmax function to normalize the weights and then sort them in descending order. Alternatively, we can use a differentiable sorting algorithm³ to directly optimize the ranking metric.
- Finally, we need to define a loss function that can measure the performance of the search engine. For example, we can use the mean reciprocal rank (MRR) or the normalized discounted cumulative gain (NDCG) as the evaluation metrics. The loss function should be differentiable with respect to the query model, the attention function, and the output model parameters.

By designing the search engine in this way, we can use gradient-based methods to optimize its components and learn from user feedback. This can potentially improve the accuracy and efficiency of the search engine and provide a better user experience. 

- (1) Att-DARTS: Differentiable Neural Architecture Search for Attention. https://ieeexplore.ieee.org/document/9207447.
- (2) 11.1. Queries, Keys, and Values — Dive into Deep Learning 1 .... https://d2l.ai/chapter_attention-mechanisms-and-transformers/queries-keys-values.html.
- (3) Differentiable Neural Architecture Search - arXiv.org. https://arxiv.org/pdf/2101.11342.
- (4) undefined. https://ieeexplore.ieee.org/servlet/opac?punumber=9200848.

# 4. Review the design of the Squeeze and Excitation Networks (Hu et al., 2018) and interpret them through the lens of the attention mechanism.

- The "excitation" operation in SENets can be interpreted as a form of channel-wise attention mechanism.
- It assigns different attention scores to each channel, effectively guiding the network to focus more on informative channels while reducing the influence of less informative ones.
- In this sense, SENets adaptively adjust the contribution of each channel to the final representation of the feature maps, similar to how attention mechanisms adjust the importance of different parts of the input sequence in natural language processing tasks.

In Squeeze and Excitation Networks (SENet), the architecture does not explicitly define key and query vectors as in traditional attention mechanisms. Instead, it indirectly learns the key, query, and value relationships through a channel-wise attention mechanism. Here's how it works:

1. **Squeeze (Global Average Pooling)**: In the "squeeze" operation, SENet performs global average pooling (GAP) on the feature maps. This operation reduces the spatial dimensions of the feature maps to 1x1 while retaining channel-wise information.

2. **Excitation (Channel-Wise Attention)**: After the "squeeze" operation, SENet introduces a small neural network (typically a fully connected layer followed by an activation function) to model the channel-wise relationships.

   - The output of this neural network is analogous to the "query" in traditional attention mechanisms. It represents the learned importance or attention scores for each channel.
   - These learned attention scores are then used to re-weight the feature maps, where the feature maps themselves can be seen as the "value" in a traditional attention mechanism.

3. **Scaling and Rescaling**: The attention scores (queries) are used to scale (excite) the feature maps channel-wise. Important channels are amplified, while less important channels are suppressed. This is the "excitation" step in SENet.

In essence, SENet does not explicitly define separate key and value vectors as traditional attention mechanisms do. Instead, it formulates the problem as learning the relationship between the original feature maps (values) and a set of learned attention scores (queries), which are then used to adjust the feature maps.

While SENet's design differs from the traditional attention mechanism, the underlying concept remains similar: it adaptively modulates the importance of different channels (features) within the feature maps, thereby improving the representational power of the neural network.

In summary, SENet's channel-wise attention mechanism learns to implicitly compute the key, query, and value relationships by using a small neural network to generate attention scores for each channel, which are then used to scale the feature maps. This approach effectively guides the network to focus on relevant channels while downweighting less relevant ones.
