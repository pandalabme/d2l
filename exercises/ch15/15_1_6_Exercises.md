```python
!jupyter nbconvert --to markdown 11_9_7_Exercises.ipynb
```

# 1. What is the computational complexity for calculating each gradient? What could be the issue if the dictionary size is huge?

The computational complexity for calculating each gradient depends on the word embedding method and the optimization technique used. For example, for the skip-gram model with negative sampling, the complexity is O(kd), where k is the number of negative samples and d is the dimension of the word vectors¹. For the continuous bag-of-words model with hierarchical softmax, the complexity is O(log(V)d), where V is the size of the vocabulary².

If the dictionary size is huge, then the word embedding methods may face some issues, such as:

- The softmax function becomes computationally expensive, as it requires summing over all the words in the vocabulary for each input word².
- The word vectors may not be expressive enough to capture all the semantic and syntactic nuances of the language, as they are constrained by a fixed dimensionality³.
- The training time and memory requirements may increase significantly, as the number of parameters grows linearly or quadratically with the vocabulary size⁴.

- (1) On the Dimensionality of Word Embedding - arXiv.org. https://arxiv.org/pdf/1812.04224.pdf.
- (2) The Ultimate Guide to Word Embeddings - neptune.ai. https://neptune.ai/blog/word-embeddings-guide.
- (3) Learning Word Embedding | Lil'Log - GitHub Pages. https://lilianweng.github.io/posts/2017-10-15-word-embedding/.
- (4) arXiv:1411.2738v4 [cs.CL] 5 Jun 2016. https://arxiv.org/pdf/1411.2738.pdf.

# 2. Some fixed phrases in English consist of multiple words, such as “new york”. How to train their word vectors? Hint: see Section 4 in the word2vec paper (Mikolov et al., 2013).

According to the word2vec paper by Mikolov et al. (2013), one way to train word vectors for fixed phrases in English is to use a simple data-driven approach to find the phrases¹. The idea is to first train a word2vec model on the original data, and then score all possible bigrams using a formula that measures how frequently the words co-occur together compared to their individual frequencies¹. The bigrams with scores above a certain threshold are then treated as single tokens in the next iteration of training¹. This process can be repeated multiple times to find longer phrases¹. For example, the phrase "new york" may be merged into a single token "new_york" after the first iteration, and then combined with other words to form phrases like "new_york_times" or "new_york_city" in subsequent iterations¹. This way, the word vectors for phrases can capture more information than the sum of their parts.

- (1) [1310.4546] Distributed Representations of Words and .... https://arxiv.org/abs/1310.4546.
- (2) arXiv:1301.3781v3 [cs.CL] 7 Sep 2013. https://arxiv.org/pdf/1301.3781.pdf.
- (3) Mikolov, T., Chen, K., Corrado, G., et al. (2013) Efficient .... https://www.scirp.org/reference/referencespapers.aspx?referenceid=2661918.
- (4) [1301.3781] Efficient Estimation of Word Representations in .... https://arxiv.org/abs/1301.3781.
- (5) undefined. https://doi.org/10.48550/arXiv.1310.4546.

# 3. Let’s reflect on the word2vec design by taking the skip-gram model as an example. What is the relationship between the dot product of two word vectors in the skip-gram model and the cosine similarity? For a pair of words with similar semantics, why may the cosine similarity of their word vectors (trained by the skip-gram model) be high?

The relationship between the dot product of two word vectors in the skip-gram model and the cosine similarity is that they are proportional to each other, as long as the word vectors are normalized to have unit length. This is because the cosine similarity is defined as the dot product divided by the product of the magnitudes, and if the magnitudes are both 1, then the cosine similarity is equal to the dot product. Therefore, maximizing the dot product is equivalent to maximizing the cosine similarity in the skip-gram model¹⁵.

For a pair of words with similar semantics, the cosine similarity of their word vectors (trained by the skip-gram model) may be high because the skip-gram model tries to maximize the probability of predicting the context words given a center word, and words that have similar meanings tend to appear in similar contexts. Therefore, the word vectors that are trained by the skip-gram model tend to capture the semantic and syntactic similarities between words, and words that are more similar will have higher cosine similarity²⁴.

- (1) Cosine similarity versus dot product as distance metrics. https://datascience.stackexchange.com/questions/744/cosine-similarity-versus-dot-product-as-distance-metrics.
- (2) nlp - Why use cosine similarity in Word2Vec when its trained .... https://stackoverflow.com/questions/54411020/why-use-cosine-similarity-in-word2vec-when-its-trained-using-dot-product-similar.
- (3) Cosine similarity - Wikipedia. https://en.wikipedia.org/wiki/Cosine_similarity.
- (4) How the dot product measures similarity - Mathematics of .... https://www.tivadardanka.com/blog/how-the-dot-product-measures-similarity/.
- (5) Measuring Similarity from Embeddings | Machine Learning .... https://developers.google.com/machine-learning/clustering/similarity/measuring-similarity.
