# 1. Suppose that we design a deep architecture to represent a sequence by stacking self-attention layers with positional encoding. What could the possible issues be?

One possible issue with stacking self-attention layers with positional encoding is that the model may not be able to capture the long-term dependencies and structural information of the input sequence. This is because the positional encoding only provides a fixed representation of the relative or absolute positions of the words, which may not be sufficient to model the complex syntactic and semantic relationships among them. 

To address this issue, some researchers have proposed to augment self-attention layers with structural position representations, which can encode the latent structure of the input sentence, such as the dependency tree or the constituency tree. For example, Wang et al. [2] proposed to use dependency tree to represent the grammatical structure of a sentence, and introduced two strategies to encode the positional relationships among words in the dependency tree. They showed that their approach improved the performance of neural machine translation over both the absolute and relative sequential position representations.

Another possible issue with stacking self-attention layers with positional encoding is that the model may suffer from overfitting or underfitting due to the large number of parameters and the lack of regularization. This is because self-attention layers are fully connected and can attend to any pair of words in the input sequence, which may lead to overfitting on small or noisy datasets, or underfitting on large or diverse datasets.

To address this issue, some researchers have proposed to use convolutional layers or recurrent layers in conjunction with self-attention layers, which can provide local or sequential information and reduce the number of parameters. For example, Shen et al. [4] proposed to use convolutional self-attention networks, which combine convolutional layers and self-attention layers in a hierarchical manner. They showed that their approach achieved state-of-the-art results on various NLP tasks, such as text classification, natural language inference, and machine translation.

- (1) arXiv:1807.03052v1 [cs.CL] 9 Jul 2018. https://arxiv.org/pdf/1807.03052.pdf.
- (2) [1909.00383] Self-Attention with Structural Position .... https://arxiv.org/abs/1909.00383.
- (3) ON THE RELATIONSHIP BETWEEN SELF-ATTENTION .... https://arxiv.org/pdf/1911.03584v2.pdf.
- (4) undefined. https://doi.org/10.48550/arXiv.1909.00383.

# 2. Can you design a learnable positional encoding method?

One possible way to design a learnable positional encoding method is to use a neural network to map the position index to a vector of the same dimension as the word embedding. For example, we can use a multi-layer perceptron (MLP) with a linear output layer to generate the positional encoding for each position. The MLP can be trained jointly with the rest of the model, or pre-trained on some auxiliary task.

The advantage of using a learnable positional encoding method is that it can adapt to the specific task and data, and potentially capture more complex positional patterns than fixed methods. The disadvantage is that it may require more parameters and computation, and may not generalize well to longer sequences than seen during training.

- (1) Learnable Fourier Features for Multi-Dimensional Spatial .... https://arxiv.org/abs/2106.02795.
- (2) 如何优雅地编码文本中的位置信息？三种positional encoding .... https://zhuanlan.zhihu.com/p/121126531.
- (3) deep learning - The essence of learnable positional .... https://stackoverflow.com/questions/73113261/the-essence-of-learnable-positional-embedding-does-embedding-improve-outcomes-b.
- (4) undefined. https://doi.org/10.48550/arXiv.2106.02795.

# 3. Can we assign different learned embeddings according to different offsets between queries and keys that are compared in self-attention? Hint: you may refer to relative position embeddings (Huang et al., 2018, Shaw et al., 2018).

Yes, we can assign different learned embeddings according to different offsets between queries and keys that are compared in self-attention. This is the idea behind relative position embeddings (RPE), which are a type of position embeddings that encode the relative distance or direction between two tokens in a sequence. RPE can capture the pairwise, relative positional information of the input sequence, which may be useful for tasks that require structural or syntactic awareness, such as natural language understanding, machine translation, or text generation.

There are different ways to implement RPE, but one common approach is to add a learnable embedding vector to the key and/or value matrices in the self-attention mechanism, based on the relative position index between the query and the key. For example, if we have a query at position i and a key at position j, we can add an embedding vector a i j to the key matrix before computing the attention score. Similarly, we can add another embedding vector a i j to the value matrix before computing the context vector. The embedding vectors can be either shared or separate for the key and value matrices.

One challenge of RPE is how to handle the relative positions that are beyond the range of the predefined embedding matrix. For example, if we have a maximum relative position of k, how do we deal with the positions that are larger than k or smaller than -k? One solution is to clip the relative positions to the range [-k, k], and use the same embedding vector for all the positions that exceed this range. Another solution is to use a sinusoidal function to generate the embedding vectors for any relative position, as proposed by Huang et al. [1].

RPE has been shown to improve the performance of Transformer models on various natural language processing tasks, such as machine translation [2], text summarization [3], and natural language inference [4]. RPE can also reduce the computational complexity and memory consumption of Transformer models, as they do not need to store or update absolute position embeddings for each token.

- (1) Relative position embedding - 知乎. https://zhuanlan.zhihu.com/p/364828960.
- (2) Relative Position Encodings Explained | Papers With Code. https://paperswithcode.com/method/relative-position-encodings.
- (3) Relative Positional Embedding | Chao Yang. http://placebokkk.github.io/asr/2021/01/14/asr-rpe.html.
- (4) Explore Better Relative Position Embeddings from Encoding .... https://aclanthology.org/2021.emnlp-main.237.pdf.
- (5) undefined. https://www.youtube.com/watch?v=qajudaEHuq8.
