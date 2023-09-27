# 1. How can we sample noise words in negative sampling?

Negative sampling is a technique used in word embedding models, such as Word2Vec, to train word vectors efficiently. In negative sampling, you select a few "negative" words (i.e., words that are not the target word) to update the model's parameters along with the "positive" word (the actual target word). These negative samples help the model distinguish between the target word and other words in the vocabulary. The selection of negative samples is done probabilistically based on word frequency.

Here's a high-level overview of how you can sample noise words (negative samples) in negative sampling:

1. **Prepare a Word Frequency Distribution**: Calculate the frequency of each word in your training corpus. You can count the number of times each word appears in the corpus.

2. **Calculate Probabilities**: Calculate the probabilities for each word to be selected as a negative sample. The probability of selecting a word as a negative sample is often proportional to its frequency. Common choices include unigram or unsmoothed probabilities.

3. **Create a Noise Distribution**: Convert the probabilities into a noise distribution. This can be achieved by normalizing the probabilities so that they sum to 1, effectively creating a probability distribution over the entire vocabulary.

4. **Sampling**: To sample a noise word during training, you can use techniques like the following:
   
   - **Uniform Sampling**: Generate a random number between 0 and 1 and select the word whose cumulative probability surpasses this random number. This approach ensures that words with higher probabilities are more likely to be selected.

   - **Alias Sampling**: Alias sampling is a more efficient method for sampling based on probabilities. It precomputes alias tables to speed up the sampling process. Alias tables can be created based on the noise distribution.

   - **Negative Sampling Table**: Another efficient approach is to create a negative sampling table. This table contains word indices, with each index repeated according to its probability. You can then randomly select an index from this table.

5. **Update Model**: Once you have selected the negative samples, use them along with the positive sample (the actual target word) to update the model's parameters (e.g., word embeddings) through gradient descent.

6. **Repeat**: Repeat this process for each training example and over multiple training iterations to optimize your word vectors.

The choice of the number of negative samples (i.e., how many negative words to sample for each positive word) is typically a hyperparameter that you can tune. Too few negative samples may not provide enough contrastive information, while too many may slow down training.

In practice, negative sampling has been widely used in word embedding models like Word2Vec to efficiently train word vectors, making them suitable for large-scale natural language processing tasks.


```python
import random

# Example word frequencies (you can replace this with your actual data)
word_frequencies = {
    "apple": 10,
    "banana": 5,
    "cherry": 3,
    "date": 2,
    "elderberry": 1
}

# Step 1: Calculate the total frequency
total_frequency = sum(word_frequencies.values())

# Step 2: Calculate the probabilities
word_probabilities = {word: freq / total_frequency for word, freq in word_frequencies.items()}

# Step 3: Create a noise distribution
def create_noise_distribution(probabilities):
    # Create a list of words and their cumulative probabilities
    cumulative_probabilities = []
    cumulative_prob = 0
    for word, prob in probabilities.items():
        cumulative_prob += prob
        cumulative_probabilities.append((word, cumulative_prob))
    
    # Return the list of words and cumulative probabilities
    return cumulative_probabilities

noise_distribution = create_noise_distribution(word_probabilities)

# Step 4: Sampling from the noise distribution
def sample_from_noise_distribution(noise_distribution):
    # Generate a random number between 0 and 1
    random_number = random.random()
    
    # Find the word whose cumulative probability surpasses the random number
    for word, cumulative_prob in noise_distribution:
        if random_number <= cumulative_prob:
            return word

# Sample some noise words
num_samples = 5
noise_samples = [sample_from_noise_distribution(noise_distribution) for _ in range(num_samples)]

print("Noise Samples:", noise_samples)

```

    Noise Samples: ['banana', 'date', 'apple', 'cherry', 'banana']


# 2. Verify that $\sum_{w\in V}{P(w|w_c)}=1$ (15.2.9) holds.

To verify that $\sum_{w\in V}{P(w|w_c)}=1$ in Hierarchical Softmax holds, we need to use the properties of the Huffman tree and the sigmoid function. 

First, we note that the Huffman tree is a full binary tree, which means that every non-leaf node has exactly two children. Therefore, for any non-leaf node $p_j^w$, the sum of the probabilities of its two children is equal to one, i.e.,

$$
p(0|\mathbf{x}_w,\theta_{j-1}^w) + p(1|\mathbf{x}_w,\theta_{j-1}^w) = 1
$$

where $\mathbf{x}_w$ is the input vector and $\theta_{j-1}^w$ is the parameter vector for node $p_j^w$. This follows from the definition of the sigmoid function:

$$
\sigma(x) = \frac{1}{1+e^{-x}}
$$

and its property:

$$
\sigma(-x) = 1 - \sigma(x)
$$

Second, we note that the probability of a word $w$ given the input vector $\mathbf{x}_w$ is the product of the probabilities along the path from the root to the leaf node corresponding to $w$, i.e.,

$$
P(w|\mathbf{x}_w) = \prod_{j=2}^{l^w} p(d_j^w|\mathbf{x}_w,\theta_{j-1}^w)
$$

where $l^w$ is the length of the path, $d_j^w$ is the binary code of node $p_j^w$, and $p(d_j^w|\mathbf{x}_w,\theta_{j-1}^w)$ is either $\sigma(\mathbf{x}_w^\top\theta_{j-1}^w)$ or $1-\sigma(\mathbf{x}_w^\top\theta_{j-1}^w)$ depending on whether $d_j^w$ is 0 or 1.

Now, we can prove that $\sum_{w\in V}{P(w|\mathbf{x}_w)}=1$ by induction on the depth of the Huffman tree. 

Base case: If the depth of the tree is 1, then there are only two words in the vocabulary, and their probabilities are:

$$
P(w_1|\mathbf{x}_{w_1}) = \sigma(\mathbf{x}_{w_1}^\top\theta_0^{w_1}) $$
$$
P(w_2|\mathbf{x}_{w_2}) = 1 - \sigma(\mathbf{x}_{w_2}^\top\theta_0^{w_2})
$$

where $\theta_0^{w_1}$ and $\theta_0^{w_2}$ are the parameter vectors for the root node. Since $\mathbf{x}_{w_1}$ and $\mathbf{x}_{w_2}$ are arbitrary vectors, we can assume without loss of generality that $\mathbf{x}_{w_1} = \mathbf{x}_{w_2} = \mathbf{x}$. Then, we have:

$$
\sum_{w\in V}{P(w|\mathbf{x})} = \sigma(\mathbf{x}^\top\theta_0^{w_1}) + 1 - \sigma(\mathbf{x}^\top\theta_0^{w_2}) \\
= \sigma(\mathbf{x}^\top\theta_0^{w_1}) + \sigma(-\mathbf{x}^\top\theta_0^{w_2}) \\
= 1
$$

where we used the property of the sigmoid function in the last step.

Inductive step: Suppose that for any Huffman tree with depth less than or equal to $k$, we have $\sum_{w\in V}{P(w|\mathbf{x}_w)}=1$. Now, consider a Huffman tree with depth $k+1$. We can divide the vocabulary into two subsets, $V_L$ and $V_R$, corresponding to the left and right subtrees of the root node. Then, we have:

$$
\sum_{w\in V}{P(w|\mathbf{x}_w)} = \sum_{w\in V_L}{P(w|\mathbf{x}_w)} + \sum_{w\in V_R}{P(w|\mathbf{x}_w)}
$$

By applying the definition of Hierarchical Softmax, we can rewrite each term as:

$$
\sum_{w\in V_L}{P(w|\mathbf{x}_w)} = p(0|\mathbf{x},\theta_0) \sum_{w\in V_L}{P(w|\mathbf{x}_w,p_0^w)} \\
\sum_{w\in V_R}{P(w|\mathbf{x}_w)} = p(1|\mathbf{x},\theta_0) \sum_{w\in V_R}{P(w|\mathbf{x}_w,p_0^w)}
$$

where $\mathbf{x}$ is the input vector, $\theta_0$ is the parameter vector for the root node, and $p_0^w$ is the root node itself. Note that $P(w|\mathbf{x}_w,p_0^w)$ is the conditional probability of word $w$ given the input vector $\mathbf{x}_w$ and the root node $p_0^w$, which is equivalent to the probability of word $w$ given the input vector $\mathbf{x}_w$ in the subtree rooted at $p_0^w$. Since the depth of each subtree is less than or equal to $k$, we can apply the induction hypothesis and obtain:

$$
\sum_{w\in V_L}{P(w|\mathbf{x}_w,p_0^w)} = 1 \\
\sum_{w\in V_R}{P(w|\mathbf{x}_w,p_0^w)} = 1
$$

Therefore, we have:

$$
\sum_{w\in V}{P(w|\mathbf{x}_w)} = p(0|\mathbf{x},\theta_0) + p(1|\mathbf{x},\theta_0) \\
= 1
$$

where we used the property of the Huffman tree in the last step.

Hence, by induction, we have proved that $\sum_{w\in V}{P(w|\mathbf{x}_w)}=1$ in Hierarchical Softmax holds for any Huffman tree with any depth.

For more information about Hierarchical Softmax, you can refer to [Hierarchical Softmax Explained](^1^), [Hierarchical Softmax（层次Softmax）](^2^), or [hierarchical softmaxについて](^3^).

- (1) Hierarchical Softmax Explained | Papers With Code. https://paperswithcode.com/method/hierarchical-softmax.
- (2) Hierarchical Softmax（层次Softmax） - 知乎. https://zhuanlan.zhihu.com/p/56139075.
- (3) hierarchical softmaxについて｜gota_morishita - note（ノート）. https://note.com/gota_morishita/n/nb9c9f126783d.

# 3. How to train the continuous bag of words model using negative sampling and hierarchical softmax, respectively?

The continuous bag of words (CBOW) model is a neural network that predicts a target word given a context of surrounding words. The model consists of an input layer, a hidden layer, and an output layer. The input layer is a one-hot vector that represents the context words, the hidden layer is a dense vector that represents the word embeddings, and the output layer is a softmax layer that predicts the probability of each word in the vocabulary being the target word.

To train the CBOW model, we need to define a loss function that measures how well the model predicts the target word given the context words. A common choice of loss function is the cross-entropy loss, which is defined as:

$$
L = -\log P(w_t|w_{t-n},\dots,w_{t-1},w_{t+1},\dots,w_{t+n})
$$

where $w_t$ is the target word, $w_{t-n},\dots,w_{t+n}$ are the context words, and $P(w_t|w_{t-n},\dots,w_{t-1},w_{t+1},\dots,w_{t+n})$ is the output of the softmax layer.

However, computing the softmax layer requires summing over all the words in the vocabulary, which can be very expensive when the vocabulary size is large. Therefore, some algorithmic optimizations are proposed to speed up the training process, such as negative sampling and hierarchical softmax.

Negative sampling is a technique that approximates the softmax layer by only considering a small number of negative samples, which are randomly chosen words that are not the target word. The idea is to train the model to distinguish the target word from the negative samples, rather than from all the other words in the vocabulary. The loss function for negative sampling is defined as:

$$
L = -\log \sigma(\mathbf{v}_{w_t}^\top \mathbf{v}'_{w_{t-n},\dots,w_{t+n}}) - \sum_{i=1}^k \mathbb{E}_{w_i \sim P_n(w)}[\log \sigma(-\mathbf{v}_{w_i}^\top \mathbf{v}'_{w_{t-n},\dots,w_{t+n}})]
$$

where $\sigma$ is the sigmoid function, $\mathbf{v}_{w_t}$ and $\mathbf{v}'_{w_{t-n},\dots,w_{t+n}}$ are the output and input embeddings of the target and context words, respectively, $k$ is the number of negative samples, and $P_n(w)$ is a noise distribution that assigns probabilities to each word in the vocabulary. A common choice of $P_n(w)$ is $P_n(w) = \frac{U(w)^{3/4}}{\sum_{w'} U(w')^{3/4}}$, where $U(w)$ is the unigram frequency of word $w$.

Hierarchical softmax is another technique that reduces the computation of the softmax layer by organizing the words in a binary tree, where each leaf node corresponds to a word in the vocabulary. The idea is to train the model to predict the path from the root node to the target word node, rather than predicting the probability of each word in the vocabulary. The loss function for hierarchical softmax is defined as:

$$
L = -\sum_{j=2}^{l^w} \log \sigma(d_j^w \mathbf{v}_{p_j^w}^\top \mathbf{v}'_{w_{t-n},\dots,w_{t+n}})
$$

where $l^w$ is the length of the path from root to target word node, $p_j^w$ is the j-th node on this path, $d_j^w$ is 1 if $p_j^w$ is a left child and -1 if it is a right child, and $\mathbf{v}_{p_j^w}$ and $\mathbf{v}'_{w_{t-n},\dots,w_{t+n}}$ are the output and input embeddings of node $p_j^w$ and context words, respectively.

For more information about these techniques, you can refer to [arXiv:1411.2738v4 [cs.CL] 5 Jun 2016](^1^), [CS224n: Natural Language Processing with Deep Learning](^2^), [Word2Vec — CBOW & Skip-gram : Algorithmic Optimizations](^3^), or [Word2Vec in Pytorch - Continuous Bag of Words and Skipgrams](^4^).

- (1) arXiv:1411.2738v4 [cs.CL] 5 Jun 2016. https://arxiv.org/pdf/1411.2738.pdf.
- (2) CS224n: Natural Language Processing with Deep Learning. https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes01-wordvecs1.pdf.
- (3) Word2Vec — CBOW & Skip-gram : Algorithmic Optimizations. https://medium.com/analytics-vidhya/word2vec-cbow-skip-gram-algorithmic-optimizations-921d6f62d739.
- (4) Word2Vec in Pytorch - Continuous Bag of Words and Skipgrams. https://srijithr.gitlab.io/post/word2vec/.
- (5) undefined. http://bit.ly/wevi-online.
