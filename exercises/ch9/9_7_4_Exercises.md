# 1. Assume that we have a symmetric matrix $M\in\mathbb{R}^{n\times m}$ with eigenvalues $\lambda_i$ whose corresponding eigenvectors are $v_i$($i=1,\dots,n$). Without loss of generality, assume that they are ordered in the order $\left|\lambda_i\right|\ge\left|\lambda_{i+1}\right|$.

## 1.1 Show that $M^k$ has eigenvalues $\lambda_i^k$.

$Mv_i=\lambda_iv_i$

$M^kv_i=M^{k-1}(Mv_i)=\lambda_iM^{k-1}v_i=\lambda_i^kv_i$

## 1.2 Prove that for a random vector $x\in\mathbb{R}^n$, with high probability $M^kx$ will be very much aligned with the eigenvector $v_1$ of $M$. Formalize this statement.

One possible way to prove the statement is to use the power method¹², which is an iterative algorithm for finding the dominant eigenvalue and eigenvector of a matrix. The idea is to start with a random vector $x_0\in\mathbb{R}^n$ and repeatedly multiply it by the matrix $M$, normalizing it after each step. That is, we define $x_{k+1}=\frac{Mx_k}{\|Mx_k\|}$ for $k=0,1,2,\dots$. The power method guarantees that, under some mild assumptions on $M$, the sequence $\{x_k\}$ will converge to a unit eigenvector of $M$ corresponding to the eigenvalue with the largest absolute value. Moreover, the convergence rate depends on the ratio of the largest and second largest eigenvalues of $M$. If this ratio is large, then the convergence is fast.

To formalize this statement, we need to define what it means for a vector to be "very much aligned" with another vector. A natural way to measure the alignment is by using the angle between them. If the angle is close to zero, then the vectors are almost parallel. If the angle is close to $\pi/2$, then the vectors are almost orthogonal. The angle between two unit vectors $u$ and $v$ can be computed by using the dot product: $\cos\theta=u\cdot v$. Therefore, we can say that $x_k$ is very much aligned with $v_1$ if $\cos\theta_k=|x_k\cdot v_1|$ is close to one.

To quantify how close $\cos\theta_k$ is to one, we can use some error tolerance $\epsilon>0$. For example, we can say that $x_k$ is $\epsilon$-aligned with $v_1$ if $\cos\theta_k\geq 1-\epsilon$. Then, we can state the following theorem:

**Theorem:** Let $M$ be an $n\times n$ matrix with a dominant eigenvalue $\lambda_1$ and a corresponding unit eigenvector $v_1$. Let $x_0$ be a random vector in $\mathbb{R}^n$ with unit norm. Let $\epsilon>0$ be an error tolerance. Then, there exists a positive integer $K$ such that for all $k\geq K$, the vector $x_k=\frac{M^kx_0}{\|M^kx_0\|}$ is $\epsilon$-aligned with $v_1$. That is, $$|x_k\cdot v_1|\geq 1-\epsilon$$

**Proof:** The proof of this theorem can be found in many textbooks on numerical analysis or linear algebra¹². Here, we only sketch the main steps. First, we write $x_0$ as a linear combination of the eigenvectors of $M$: $$x_0=c_1v_1+c_2v_2+\cdots+c_nv_n$$ where $c_i\neq 0$ for some $i>1$. Then, we multiply both sides by $M^k$ and use the fact that $Mv_i=\lambda_iv_i$ for each $i$: $$M^kx_0=c_1\lambda_1^kv_1+c_2\lambda_2^kv_2+\cdots+c_n\lambda_n^kv_n$$ Next, we divide both sides by $\lambda_1^k$ and use the fact that $\lambda_1$ is the dominant eigenvalue: $$\frac{M^kx_0}{\lambda_1^k}=c_1v_1+c_2(\frac{\lambda_2}{\lambda_1})^kv_2+\cdots+c_n(\frac{\lambda_n}{\lambda_1})^kv_n$$ Since $\left|\frac{\lambda_i}{\lambda_1}\right|<1$ for all $i>1$, the terms involving $\lambda_i^k$ will decay exponentially as $k$ increases. Therefore, for sufficiently large $k$, we have $$\frac{M^kx_0}{\lambda_1^k}\approx c_1v_1$$ Finally, we normalize both sides by their norms and use the fact that $\|v_1\|=1$: $$x_k=\frac{M^kx_0}{\|M^kx_0\|}\approx \frac{c_1v_1}{|c_1|\|v_1\|}= \pm v_1$$ This implies that $$|x_k\cdot v_1|\approx |(\pm v_1)\cdot v_1|= 1$$ Hence, for any given $\epsilon>0$, we can find a positive integer $K$ such that for all $k\geq K$, we have $$|x_k\cdot v_1|\geq 1-\epsilon$$ This completes the proof. $\blacksquare$


- (1) Derivation of power method - Mathematics Stack Exchange. https://math.stackexchange.com/questions/939148/derivation-of-power-method.
- (2) 10.3 POWER METHOD FOR APPROXIMATING EIGENVALUES. https://ergodic.ugr.es/cphys/lecciones/fortran/power_method.pdf.
- (3) Power Method Proof - Il metodo delle potenze è un semplice .... https://www.studocu.com/it/document/universita-telematica-e-campus/analisi-numerica/power-method-proof-il-metodo-delle-potenze-e-un-semplice-metodo-iterativo-per-il-calcolo-approssimato/16444461.
- (4) Power iteration - Wikipedia. https://en.wikipedia.org/wiki/Power_iteration.

## 1.3 What does the above result mean for gradients in RNNs?

The result mentioned, which deals with the alignment of vectors under repeated matrix multiplication, has some relevance to the gradients in Recurrent Neural Networks (RNNs), especially in the context of gradient vanishing and exploding problems.

In RNNs, you have sequences of vectors, and you often perform repeated matrix-vector multiplications (hidden state updates) at each time step. These matrix-vector multiplications are analogous to the repeated matrix powers in the result you mentioned. Here's how the result might relate to gradients in RNNs:

1. **Gradient Vanishing**: In some RNN architectures, particularly vanilla RNNs, the gradients can vanish when backpropagating through time (BPTT). This means that as you backpropagate through multiple time steps, the gradients can become extremely small. The result you mentioned about alignment is somewhat related because it highlights that when you repeatedly multiply by a matrix with eigenvalues less than 1, the vector becomes aligned with the dominant eigenvector and can converge to zero. In the context of RNNs, this can lead to vanishing gradients, making it challenging to train the network effectively.

2. **Gradient Exploding**: On the flip side, if the matrix you repeatedly multiply by has eigenvalues greater than 1, the result can explode, leading to large values. In the context of RNNs, this corresponds to exploding gradients. This is also a problem because it can lead to numerical instability during training.

3. **Solutions**: To address these issues, various RNN architectures and optimization techniques have been proposed. LSTMs and GRUs, for example, have gating mechanisms that help mitigate gradient vanishing to some extent. Additionally, techniques like gradient clipping and using well-conditioned weight matrices can help prevent gradient explosion.

4. **Weight Initialization**: Weight initialization schemes, such as the Xavier/Glorot initialization, are designed to set the initial weight matrices of RNNs in a way that controls the eigenvalues and helps mitigate both vanishing and exploding gradients.

5. **Orthogonal Initialization**: In some cases, orthogonal weight initialization, which initializes the weight matrices as orthogonal matrices, can help maintain the alignment of vectors during repeated multiplications and mitigate the vanishing gradient problem.

In summary, the result about vector alignment under repeated matrix multiplication has implications for understanding the challenges of gradient vanishing and exploding in RNNs. Addressing these challenges is crucial for training RNNs effectively, and various architectural and initialization techniques have been developed to manage these issues.

# 2. Besides gradient clipping, can you think of any other methods to cope with gradient explosion in recurrent neural networks?

In addition to gradient clipping, there are several other methods and techniques to cope with gradient explosion in recurrent neural networks (RNNs):

1. **Weight Regularization**:
   - **L2 Weight Regularization (Weight Decay)**: Adding an L2 regularization term to the loss function encourages weight values to be small, which can help prevent gradients from exploding.
   - **L1 Weight Regularization**: Similar to L2 regularization, L1 regularization encourages sparsity in weight matrices and can help control gradient explosion.

2. **Batch Normalization**:
   - Applying batch normalization to hidden state activations can help stabilize training by normalizing activations within each mini-batch. While batch normalization is often used in feedforward networks, it can also be adapted for use in recurrent layers.

3. **Gradient Clipping Variations**:
   - **Norm-based Clipping**: Instead of clipping gradients element-wise, you can clip the norm (e.g., using the L2 norm) of the entire gradient vector. This ensures that the gradient direction is maintained but its magnitude is controlled.

4. **Truncated Backpropagation Through Time (TBPTT)**:
   - Instead of backpropagating through the entire sequence, you can limit the backpropagation to a fixed number of time steps. This reduces the number of matrix multiplications and can mitigate gradient explosion.

5. **Gradient Checkpointing**:
   - This technique involves checkpointing intermediate activations during the forward pass and recomputing them during the backward pass. It can help reduce memory consumption during backpropagation through time, which can indirectly address gradient explosion.

6. **Exploding Gradient Detection**:
   - Implement mechanisms to detect when gradients are exploding during training. For example, you can monitor the norm of the gradient during training and apply gradient clipping or other techniques when it exceeds a threshold.

7. **Gradient Scaling**:
   - You can apply a scaling factor to gradients, reducing their magnitude. This effectively dampens the effect of large gradients while preserving their direction.

8. **Curriculum Learning**:
   - Gradually increasing the sequence length during training can help networks learn to handle longer sequences more effectively and potentially reduce gradient explosion.

9. **Alternative Architectures**:
   - Consider using more stable RNN architectures, such as Long Short-Term Memory (LSTM) or Gated Recurrent Unit (GRU), which are designed to mitigate the vanishing and exploding gradient problems to some extent.

10. **Gradient Clipping on a Per-Layer Basis**:
    - Instead of applying a uniform gradient clipping threshold to all layers, you can apply different clipping thresholds to different layers in the network. This allows for finer control over the gradient explosion problem in specific parts of the network.

11. **Learning Rate Scheduling**:
    - Reduce the learning rate over time during training. Gradually decreasing the learning rate can help stabilize training and reduce the likelihood of gradient explosion as the optimization process converges.

12. **Gradient Masking**:
    - In some cases, you might selectively mask out certain gradients or gradients from specific layers to prevent them from contributing to gradient explosion.

Experimentation and model-specific considerations are often necessary to determine the most effective approach to coping with gradient explosion in RNNs, as the effectiveness of these methods can vary depending on the specific problem and architecture.
