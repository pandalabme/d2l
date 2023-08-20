# 4.6.5 Exercises

## 1. If we wish to estimate the error of a fixed model f to within 0.0001 with probability greater than 99.9%, how many samples do we need?

Use Hoeffding's inequality
$$P\left(X_i - \mathbb{E}[X_i] \geq \epsilon \right) \leq \exp\left(-2n\epsilon^2\right)$$
We can get the number of samples we need is nearly 350 million.


```python
import numpy as np
def stat_n(p, t):
    return -np.log(1-p)/2/t**2

print(f'{stat_n(0.999, 0.0001):.2g}')
```

    3.5e+08


## 2. Suppose that somebody else possesses a labeled test set D and only makes available the unlabeled inputs (features). Now suppose that you can only access the test set labels by running a model f (with no restrictions placed on the model class) on each of the unlabeled inputs and receiving the corresponding error $\epsilon_D(f)$. How many models would you need to evaluate before you leak the entire test set and thus could appear to have error 0, regardless of your true error?

This scenario is related to the concept of overfitting to the test set, which can lead to a phenomenon known as "test set overfitting" or "data leakage." Essentially, by repeatedly evaluating a model on the test set and using the test set errors to adjust the model, you risk unintentionally fitting the model to the test set and obtaining misleadingly optimistic results, there isn't a specific number of models that you need to evaluate before you "leak" the entire test set and risk overfitting.

If you continue evaluating different models on the unlabeled test inputs and using their corresponding test set errors to adjust the models, you can potentially find a model that matches the test set perfectly. However, this does not reflect the true generalization performance of the model to unseen data.

In other words, if you keep adjusting and selecting models based on their performance on the test set, you might end up with a model that perfectly fits the test set but performs poorly on new, unseen data. This is a form of overfitting that is specific to the test set and does not reflect the model's ability to generalize to other data.

## 3. What is the VC dimension of the class of fifth-order polynomials?

Using the formula of linear models on d-dimensional inputs have VC dimension **d+1**, we can get the VC dimension of the class of fifth-order polynomials is **6**. This means that you can find six points in the input space that can be shattered by the set of fifth-order polynomials, but it's not possible to shatter any set of seven points. In other words, you can find six points and construct a set of polynomial functions that can perfectly separate any possible labeling of those points, but for any set of seven points, there will be at least one labeling that cannot be achieved by any fifth-order polynomial function.

## 4. What is the VC dimension of axis-aligned rectangles on two-dimensional data?

The VC (Vapnik-Chervonenkis) dimension of axis-aligned rectangles on two-dimensional data is 4. 

In the context of classification problems, axis-aligned rectangles are binary classifiers that split the input space into two regions based on the position of a rectangle aligned with the axes. The VC dimension of this hypothesis class represents the maximum number of points in general position that can be shattered by these axis-aligned rectangles.

For axis-aligned rectangles in two dimensions, you can find a set of 4 points that can be shattered by this hypothesis class, but any set of 5 points cannot be shattered. This means that you can label a set of 4 points in such a way that you can find an axis-aligned rectangle that perfectly separates them, but for any set of 5 points, there will be at least one labeling that cannot be achieved by any combination of axis-aligned rectangles.

reference: https://www.cs.princeton.edu/courses/archive/spring13/cos511/scribe_notes/0221.pdf
