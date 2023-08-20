# 4.1.5. Exercises

## 1. We can explore the connection between exponential families and softmax in some more depth.
* Compute the second derivative of the cross-entropy loss $l(y,\hat{y})$ for softmax.
* Compute the variance of the distribution given by $\text{softmax}(o)$ and show that it matches the second derivative computed above.

### Second derivative of the cross-entropy loss

The cross-entropy loss for softmax is given by:

$$l(y, \hat{y}) = -\sum_i y_i \log(\text{softmax}(o_i))$$

where $\text{softmax}(o)_i = \frac{e^{o_i}}{\sum_j e^{o_j}}$.

Let's denote the loss as $L = -\sum_i y_i \log(\text{softmax}(o_i))$.

**First Derivative:**
The first derivative of $L$ with respect to $o_j$ is:

$$\frac{\partial L}{\partial o_j} = \text{softmax}(o_j) - y_j$$

**Second Derivative:**
The second derivative of $L$ with respect to $o_j$ is:

$$\frac{\partial^2 L}{\partial o_j^2} = \frac{\partial}{\partial o_j} (\text{softmax}(o_j) - y_j) = \frac{\partial \text{softmax}(o_j)}{\partial o_j} = \text{softmax}(o_j )(1 - \text{softmax}(o_j))$$

### Variance of the distribution

The distribution given by $softmax(𝑜)$ is actually a Bernoulli distribution with probabilities $p = softmax(o)$, so the variance is:
$$Var[X] = E[X^2] - E[X]^2 = \text{softmax}(o)(1 - \text{softmax}(o))$$

## 2. Assume that we have three classes which occur with equal probability, i.e., the probability vector is $(\frac{1}{3},\frac{1}{3},\frac{1}{3})$
* What is the problem if we try to design a binary code for it?
* Can you design a better code? Hint: what happens if we try to encode two independent observations? What if we encode observations jointly?

### 2.1 Problem

If we design a binary code directly for the three classes with equal probabilities, we might assign binary codes 00, 01, and 10 to the classes. However, this approach doesn't utilize the information about the equal probabilities effectively. It assigns different lengths to the codes, which could lead to suboptimal performance in certain scenarios.

### 2.2 Better code

We can use one-hot encode to express the jointly observations.

## 3. When encoding signals transmitted over a physical wire, engineers do not always use binary codes. For instance, PAM-3 uses three signal levels $\{-1,0,1\}$ as opposed to two levels $\{0,1\}$. How many ternary units do you need to transmit an integer in the range $\{0,...,7\}$? Why might this be a better idea in terms of electronics?

The number of ternary units needed can be found using the formula:
$ \text{Number of ternary units} = \log_3 (\text{Range}) + 1$

So in this case:
$\text{Number of ternary units} = \log_3 (7+1)  + 1 = \log_3 8 + 1 = 2$

This means that you would need just two ternary digit to represent integers in the range \(\{0, \ldots, 7\}\) using the \(-1, 0, 1\) encoding.

Using ternary encoding can be a better idea in terms of electronics for a few reasons:

1. **Increased Information Density:** Ternary encoding allows you to represent more information in a single symbol compared to binary encoding. This means that you can transmit more data in the same amount of time.

2. **Reduced Transmission Errors:** Ternary encoding with three signal levels (\(-1, 0, 1\)) can provide better noise immunity compared to binary encoding (\(0, 1\)). The presence of a zero level allows for better differentiation between signal states, reducing the likelihood of errors due to noise.

3. **Simpler Hardware:** Ternary encoding can sometimes simplify hardware design. For instance, in differential signaling systems, where the difference between signal levels matters more than the absolute values, ternary encoding can offer benefits.
4. **Easy Accomplish:** There are distinctive conditions: positive voltage, negative voltage, zero voltage in a physical wire which can be used as ternary.

## 4. The Bradley–Terry model uses a logistic model to capture preferences. For a user to choose between apples and oranges one assumes scores $o_{apple}$ and $o_{orange}$. Our requirements are that larger scores should lead to a higher likelihood in choosing the associated item and that the item with the largest score is the most likely one to be chosen (Bradley and Terry, 1952).
* Prove that softmax satisfies this requirement.
* What happens if you want to allow for a default option of choosing neither apples nor oranges? Hint: now the user has three choices.

### 4.1 Prove

In the Bradley-Terry model, the probability of item $i$ being chosen over item $j$ is given by:
$$P(i > j) = \frac{p_i}{p_i+p_j}$$
as $p_i=\text{softmax}(o_i) = \frac{e^{o_i}}{\sum_i e^{o_i}}$ and $p_i+p_j=1$, this can be simplified to:
$$P(i>j)=p_i=softmax(o_i)= \frac{e^{o_i}}{\sum_i e^{o_i}}\propto{e^{o_i}}\propto{o_i}$$
This show that larger scores lead to higher likelihood.

### 4.2 Three choices

No matter how many choices we have, the probability of choosing item i is:
$$p_i=softmax(o_i)= \frac{e^{o_i}}{\sum_i e^{o_i}}\propto{e^{o_i}}\propto{o_i}$$
Item with the largest score is the most likely one to be chosen.

## 5. Softmax gets its name from the following mapping: $RealSoftMax(a,b)=\log(\exp(a)+\exp(b))$. 
* Prove that $RealSoftMax(a,b)\gt\max(a,b)$
* How small can you make the difference between both functions? Hint: without loss of generality you can set $b=0$ and $a\geq{b}$
* Prove that this holds for $\lambda^{-1}RealSoftMax(\lambda{a},\lambda{b})$, provided that $\lambda \ge{0}$.
* Show that for $\lambda\to\infty$ we have $\lambda^{-1}RealSoftMax(\lambda{a},\lambda{b})\to\max(a,b)$.
* Construct an analogous softmin function.
* Extend this to more than two numbers.

### 5.1 Proving $RealSoftMax(a,b) > \max(a,b)$

As:
$$\exp(a) + \exp(b) > \exp(\max(a,b)) > 0$$
and log is a monotonically increasing function, we can get:
$$RealSoftMax(a,b)=log(\exp(a) + \exp(b))>\max(a,b)$$

## 5.2 Minimizing the Difference Between Functions

If we set $b = 0$ and \(a \geq b\), the functions become:
$$RealSoftMax(a, 0) = \log(\exp(a) + 1) $$
$$\max(a, 0) = a$$
So the difference between the two functions is:
$$\text{diff(a)}=\log(1+\exp(-a))$$
As $a$ increases, $\text{diff}$ gets smaller, but it's never exactly zero because $\exp(-a)$ will always be slightly larger than $0$.

### 5.3 Proving the Property Holds for Scaled Inputs

As $\lambda\gt0$:
$$\exp(\lambda{a}) + \exp(\lambda{b}) > \exp(\lambda\max(a,b)) > 0$$
and log is a monotonically increasing function, we can get:
$$RealSoftMax(a,b)=log(\exp(\lambda{a}) + \exp(\lambda{b}))>\lambda\max(a,b)$$
so:
$$\lambda^{-1}RealSoftMax(a,b)>\max(a,b)$$

### 5.4 Limit as $\lambda\to\infty$


If we set $b = 0$ and \(a \geq b\), the functions become:
$$\lambda^{-1}RealSoftMax(\lambda{a}, 0) = \log(\exp(\lambda{a}) + 1) $$
$$\max(a, 0) = a$$
So the difference between the two functions is:
$$\text{diff}(\lambda)=\lambda^{-1}\log(1+\exp(-\lambda{a}))$$
Using Lagrange's theorem:
$$\lim_{\lambda\to\infty}\text{diff}(\lambda)=\lim_{\lambda\to\infty}\frac{-1}{1+\exp(\lambda{a})}=0$$
So we have $\lambda^{-1}RealSoftMax(\lambda{a},\lambda{b})\to\max(a,b)$.

### 5.5. Analogous Softmin Function

An analogous function to the $RealSoftMax$ function can be defined as follows:
$$RealSoftMin(a, b) = -\log(\exp(-a) + \exp(-b)) $$

This function captures the "soft" version of the minimum operation, where the logarithmic function is used to create a smooth transition between the two values.



### 5.6. Extension to More Than Two Numbers
The concept can be extended to more than two numbers using a similar approach. Given numbers $a_1, a_2, \ldots, a_n$, you can define the $RealSoftMax$ as:
$$ RealSoftMax(a_1, a_2, \ldots, a_n) = \log(\sum_{i=1}^{n} \exp(a_i))$$

This function smoothens the maximum operation over multiple numbers. An analogous function, \(RealSoftMin\), can also be defined for the "soft" version of the minimum operation over more than two numbers.

## 6. The function $g(x) \overset{\mathrm{def}}{=}\log\sum{\exp{x_i}}  $ is sometimes also referred to as the log-partition function.
* Prove that the function is convex. Hint: to do so, use the fact that the first derivative amounts to the probabilities from the softmax function and show that the second derivative is the variance.
* Show that $g$ is translation invariant, i.e.$g(x+b)=g(x)$.
* What happens if some of the coordinates $x_i$ are very large? What happens if they’re all very small?
* Show that if we choose $b=\max_i{x_i}$ we end up with a numerically stable implementation.

### 6.1 Proving Convexity

To prove that the function $g(x) = \log\sum{\exp(x_i)}$ is convex, we can analyze its second derivative. First, let's find the first and second derivatives of $g(x)$:

The first derivative:
$$ \frac{d}{dx_i} g(x) = \frac{\exp(x_i)}{\sum{\exp(x_i)}}=\text{softmax}(x_i) $$

The second derivative:
$$ \frac{d^2}{dx_i^2} g(x) = \text{softmax}(x)(1-\text{softmax(x)}) \gt0$$

The second derivative is constant and non-negative, which means that the function is convex.


### 6.2 Translation Invariance??

$$g(x+b)=\log\sum\exp(x_i+b)=\log\sum\exp(x_i)\exp(b)=\log(\exp(b)\sum\exp(x_i))=\log\sum\exp(x_i)+b=g(x)+b$$

### 6.3 Behavior of \(x_i\)

- If some $x_i$ are very large, then the corresponding $\exp(x_i)$ terms dominate the sum, causing the sum to become very large. This can result in numerical instability when computing the softmax function due to limited precision in computer arithmetic.
- If all $x_i$ are very small, then the corresponding $\exp(x_i)$ terms become close to 1, resulting in a sum that is close to the number of terms. This can lead to numerical instability as well.

### 6.4 Numerically Stable Implementation

If we choose $b = \max_i{x_i}$, we can rewrite $g(x)$ as:
$$ g(x) = \log\sum{\exp(x_i)} = \log\left(\exp(b) \cdot \sum{\exp(x_i - b)}\right) = b + \log\sum{\exp(x_i - b)} $$
This form ensures that the largest value, \(b\), is subtracted from all \(x_i\), reducing the potential for numerical instability due to large exponentials. This is often used in practice to improve the numerical stability of computing $g(x)$.

## 7. Assume that we have some probability distribution $P$. Suppose we pick another distribution $Q$ with $Q(i)\propto P(i)^\alpha$ for $\alpha\gt0$.
* Which choice of $\alpha$ corresponds to doubling the temperature? Which choice corresponds to halving it?
* What happens if we let the temperature approach 0?
* What happens if we let the temperature approach $\infty$?

When we talk about adjusting the "temperature" of a probability distribution, we are referring to a concept commonly used in the context of the softmax function, especially in machine learning and optimization. The softmax function is often used to transform a vector of real values into a probability distribution. The parameter $T$, also referred to as "temperature," controls the shape of the resulting distribution.

### 7.1 Doubling and Halving the Temperature


The softmax function is defined as $Q(i) = \frac{e^{x_i/T}}{\sum_j e^{x_j/T}}$, where $x_i$ are the input values and $T$ is the temperature parameter.
As $Q(i)\propto P(i)^\alpha$ for $\alpha\gt0$, we can get:
$$\frac{Q(i)}{Q(j)}=(\frac{P(i)}{P(j)})^\alpha=e^{\frac{(x_i-x_j)}{T}}$$
This implies that the parameter $\alpha$ is related to the inverse of temperature:
$$\alpha \propto \frac{1}{T}$$
- Doubling the temperature: $\alpha$ halve. 
- Halving the temperature: $\alpha$ double.

### 7.2 Approaching Temperature of 0

As the temperature parameter $\alpha$ approaches 0, the softmax function approaches a step function. In the limit as $\alpha$ goes to 0, the softmax function will assign all probability mass to the maximum element and zero probability to the other elements. This is because as $\alpha$ becomes very small, the exponential term with the largest value will dominate the denominator, and all other terms will approach 0.

### 7.3 Approaching Temperature of $\infty$

As the temperature parameter $\alpha$ approaches $\infty$, the softmax function becomes more uniform, and the probabilities for all elements tend to converge towards equal values. In this case, the output distribution approaches a uniform distribution, where all elements have roughly the same probability.

In summary, adjusting the temperature parameter $T$ in the softmax function affects the shape and concentration of the resulting probability distribution. Higher values of $T$ "soften" the distribution, making it more uniform, while lower values "sharpen" the distribution, emphasizing the maximum value. As $T$ approaches 0, the distribution becomes more focused on the maximum value, and as $T$ approaches $\infty$, the distribution becomes more uniform.
