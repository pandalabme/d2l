# 1. As an example, there are about $3\times 10^8$ possible 6-grams in English. What is the issue when there are too many subwords? How to address the issue? Hint: refer to the end of Section 3.2 of the fastText paper (Bojanowski et al., 2017).

To address the issue, the fastText paper proposes to use a hashing function to map each subword to a fixed-size set of buckets, and use the bucket index as the input feature. This way, the dimensionality of the input matrix is reduced to the number of buckets, which is much smaller than the number of possible subwords. Furthermore, the paper suggests to use a hierarchical softmax function to speed up the output computation, and to use a sub-sampling scheme to discard the most frequent subwords, which are less informative

# 2. How to design a subword embedding model based on the continuous bag-of-words model?

Define a CBOW model that takes a sequence of context subwords as input, and outputs a probability distribution over the target subwords

# 3. To get a vocabulary of size m, how many merging operations are needed when the initial symbol vocabulary size is n?

n-m

# 4. How to extend the idea of byte pair encoding to extract phrases?

To extend the idea of BPE to extract phrases, one possible approach is to apply BPE on the word level, rather than the character level. That is, instead of treating each character as a symbol, we can treat each word as a symbol, and merge the most frequent pair of words into a new phrase. This way, we can obtain a vocabulary of phrases that capture the collocations and idioms of the language.
