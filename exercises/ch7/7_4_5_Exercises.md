# 1. Assume that we have two convolution kernels of size $k_1$ and $k_2$, respectively (with no nonlinearity in between).

## 1.1 Prove that the result of the operation can be expressed by a single convolution.


```python
K_2 = torch.normal(0, 1, (3, 3), requires_grad=True)
train_Y = corr2d(corr2d(X,K),K)
lr = 1e-3  # Learning rate

for i in range(10):
    Y_hat = corr2d(X,K_2)
    l = (Y_hat - train_Y) ** 2
    # print(l)
    
    l.sum().backward()
    # print(K_2.grad)
    # Update the kernel
    K_2.data -= lr * K_2.grad
    K_2.grad.zero_()
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')
    # print(1,K_2)
K_2
```

    epoch 2, loss 9.327
    epoch 4, loss 0.143
    epoch 6, loss 0.002
    epoch 8, loss 0.000
    epoch 10, loss 0.000





    tensor([[ 1.7815,  0.2595,  0.7211],
            [ 0.4847, -1.4443, -0.1710],
            [ 0.4958,  1.8198,  0.0529]], requires_grad=True)




```python
import torch
import torch.nn.functional as F
def corr2d(X,K):
    h,w = K.shape
    Y = torch.zeros(X.shape[0]-h+1, X.shape[1]-w+1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w]*K).sum()
    return Y

X = torch.randn(size=(8, 8))
# K_1 = torch.tensor([[1,0],[0,1]])
K_1 = torch.randn(2, 2)
K_1_1 = torch.randn(3, 3)
# print(K_1)
# print(corr2d(corr2d(X,K_1),K_1))
# K_2 = torch.tensor([[4/3,0,0],[0,4/3,0],[0,0,4/3]])
K_2 = F.conv_transpose2d(K_1.reshape(1,1,K_1.shape[0],-1), K_1_1.reshape(1,1,K_1_1.shape[0],-1)).squeeze()
print(K_1.shape,K_1_1.shape,K_2.shape)
print((corr2d(X,K_2)-corr2d(corr2d(X,K_1),K_1_1)< 1e-6).all())
```

    torch.Size([2, 2]) torch.Size([3, 3]) torch.Size([4, 4])
    tensor(True)


## 1.2 What is the dimensionality of the equivalent single convolution?

$(k_1.h+k_2.h-1,k_1.w+k_2.w-1)$

## 1.3 Is the converse true, i.e., can you always decompose a convolution into two smaller ones?

The loss may not be zero, so we can conclude that we can't always decompose a convolution into two small ones


```python
K_1 = torch.randn(2,2)
train_Y = torch.randn(3,3)
K_1_1 = torch.randn(2,2)
K_1_1.requires_grad=True
lr = 3e-2  # Learning rate
for i in range(25):
    Y_hat = F.conv_transpose2d(K_1.reshape(1,1,K_1.shape[0],-1), K_1_1.reshape(1,1,K_1_1.shape[0],-1)).squeeze()
    l = (Y_hat - train_Y) ** 2
    # print(l)
    l.sum().backward()
    # print(K_2.grad)
    # Update the kernel
    K_1_1.data -= lr * K_1_1.grad
    K_1_1.grad.zero_()
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')
    # print(1,K_2)
K_1_1
```

    epoch 2, loss 19.726
    epoch 4, loss 9.190
    epoch 6, loss 7.035
    epoch 8, loss 6.546
    epoch 10, loss 6.427
    epoch 12, loss 6.397
    epoch 14, loss 6.389
    epoch 16, loss 6.387
    epoch 18, loss 6.387
    epoch 20, loss 6.387
    epoch 22, loss 6.387
    epoch 24, loss 6.387





    tensor([[-0.8742,  0.6352],
            [-0.4420,  0.4009]], requires_grad=True)



# 2. Assume an input of shape $c_i\times h\times w$ and a convolution kernel of shape $c_o\times c_i\times k_h\times k_w$, padding of $(p_h,p_w)$, and stride of $(s_h,s_w)$.

## 2.1 What is the computational cost (multiplications and additions) for the forward propagation?

$c_o\times c_i\times k_h\times k_w\times [(h+p_h-k_h+s_h)//s_h]\times [(w+p_w-k_w+s_w)//s_w]$

## 2.2 What is the memory footprint?

To get an estimate of the memory footprint in practice, you can use the following formula:

$$
\text{Memory Footprint} = (\text{Input Data Size} + \text{Kernel Parameters Size} + \text{Output Data Size} + \text{Activation Maps Size}) \times \text{Data Type Size}
$$

Keep in mind that this is a simplified estimate and might not account for all memory overheads in the deep learning framework and hardware.

## 2.3 What is the memory footprint for the backward computation?

The total memory footprint for the backward computation is the sum of memory used by these factors. To get an estimate of the memory footprint in practice, you can use a similar formula as for the forward pass:

$$
\text{Memory Footprint} = (\text{Gradients Size} + \text{Intermediate Values Size} + \text{Activation Maps Size} + \text{Workspace Size}) \times \text{Data Type Size}
$$

It's important to note that deep learning libraries and hardware might optimize memory usage differently for forward and backward passes. Additionally, memory consumption can vary based on the specific optimization settings, hardware architecture, and implementation details.

## 2.4 What is the computational cost for the backpropagation?

The computational cost for the backpropagation (gradient computation) of a convolutional layer is typically higher than that of the forward propagation. This is because backpropagation involves computing gradients with respect to both the layer's input and its parameters (weights and biases), which requires additional calculations.

The computational cost for the backpropagation of a convolutional layer includes the following main factors:

1. Gradients with Respect to Output: Computing gradients with respect to the output of the layer (output feature map) involves similar calculations to the forward propagation. The number of multiplications and additions for each output element is determined by the size of the kernel and input region.

2. Gradients with Respect to Weights: Computing gradients with respect to the weights (kernel parameters) involves multiplying the gradients from the output layer with the corresponding input values. The number of multiplications and additions depends on the kernel size and input region.

3. Gradients with Respect to Input: Computing gradients with respect to the input requires reversing the convolution operation, which involves transposing the kernel and applying it to the gradients from the output layer. This operation has a similar computational cost to the forward convolution.

The total computational cost for the backpropagation of the convolutional layer can be calculated by summing up the calculations required for these factors. Keep in mind that the actual calculations may involve additional operations for memory management, gradient updates, and optimization-related computations.

In general, the backpropagation cost is higher than the forward propagation cost due to the need to compute gradients and propagate them through the network. However, modern deep learning libraries and hardware accelerate these operations, and optimizations like gradient aggregation and weight updates might further impact the overall computational cost.

# 3. By what factor does the number of calculations increase if we double both the number of input channels $c_i$ and the number of output channels $c_o$? What happens if we double the padding?

As the formula of computational cost (multiplications and additions) for the forward propagation is 
$$c_o\times c_i\times k_h\times k_w\times [(h+p_h-k_h+s_h)//s_h]\times [(w+p_w-k_w+s_w)//s_w]$$
We can conclude that the number of calculations will be **4 times** if we double both number of input and output channels, while double padding will not increase that much.

# 4. Are the variables Y1 and Y2 in the final example of this section exactly the same? Why?

Not exactly the same, because the theoretical equivalence holds when the operations are performed with ideal precision and mathematical properties. In practice, especially with finite precision computations, slight differences may arise due to numerical limitations and implementation details.


```python
def corr2d_multi_in(X, K):
    # Iterate through the 0th dimension (channel) of K first, then add them up
    return sum(corr2d(x, k) for x, k in zip(X, K))

def corr2d_multi_in_out(X, K):
    # Iterate through the 0th dimension of K, and each time, perform
    # cross-correlation operations with input X. All of the results are
    # stacked together
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # Matrix multiplication in the fully connected layer
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))

X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
print(float(torch.abs(Y1 - Y2).sum()) < 1e-6)
(Y1 == Y2).all()
```

    True





    tensor(False)



# 5. Express convolutions as a matrix multiplication, even when the convolution window is not $1\times 1$.


```python
def corr2d_matmul(pic, K):
    pad_K = F.pad(K,(0,pic.shape[1]-K.shape[1],0,pic.shape[0]-K.shape[0])).type(torch.float32)
    l = []
    for i in range(pic.shape[0]-K.shape[0]+1):
        for j in range(pic.shape[1]-K.shape[1]+1):
            l.append(torch.roll(pad_K,(i,j),(0,1)).reshape(1,-1))
    print(torch.cat(l,dim=0))
    return (torch.cat(l,dim=0)@pic.reshape(-1,1)).reshape(pic.shape[0]-K.shape[0]+1,pic.shape[1]-K.shape[1]+1)

K = torch.randn(1,1)
x = torch.randn(3,3)
(corr2d_matmul(x,K) == corr2d(x,K)).all()
```

    tensor([[0.6307, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.6307, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.6307, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.6307, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.6307, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.6307, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.6307, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.6307, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.6307]])





    tensor(True)



# 6. Your task is to implement fast convolutions with a $k\times k $kernel. One of the algorithm candidates is to scan horizontally across the source, reading a $k$-wide strip and computing the $1$-wide output strip one value at a time. The alternative is to read a $k+\Delta$ wide strip and compute a $\Delta$-wide output strip. Why is the latter preferable? Is there a limit to how large you should choose $\Delta$?

The alternative approach of reading a $k + \Delta$ wide strip and computing a $\Delta$-wide output strip is preferable for several reasons, and it can lead to more efficient convolution operations. This approach is often known as "vectorized convolution" or "strip mining."

Advantages of the Alternative Approach:
1. **Memory Access Pattern**: The alternative approach increases the data reuse by reading a larger strip of input data ($k + \Delta$ wide) at once. This can take better advantage of memory hierarchy and cache, reducing the number of memory accesses compared to reading $k$-wide strips multiple times.

2. **Parallelization**: The wider strip allows for more opportunities for parallelism since you can perform computations for multiple output values simultaneously. This can be beneficial for optimizing computations on modern hardware, like GPUs or parallel CPU architectures.

3. **Reduced Overhead**: Reading a larger strip of input data with fewer iterations reduces loop overhead and branching, which can lead to improved performance.

However, there is a limit to how large you should choose $\Delta$, and this depends on factors such as memory constraints, cache size, and hardware architecture. Increasing $\Delta$ too much can lead to increased memory usage and cache thrashing, negating the benefits of the approach. A larger $\Delta$ also requires more computation to process the wider strip, which might introduce additional overhead.

It's important to find a balance between the benefits of increased data reuse and parallelism and the potential drawbacks of increased memory usage and computation. Profiling and experimenting with different values of $\Delta$ on the specific hardware you're targeting can help you determine an optimal choice for $\Delta$.

In summary, the alternative approach of reading a $k + \Delta$ wide strip and computing a $\Delta$-wide output strip is preferable due to improved memory access patterns, increased parallelism, and reduced overhead. However, the choice of $\Delta$ should be carefully considered based on hardware limitations and performance trade-offs.

# 7. Assume that we have a $c\times c$ matrix.

## 7.1 How much faster is it to multiply with a block-diagonal matrix if the matrix is broken up into $b$ blocks?

When you multiply a matrix with a block-diagonal matrix, the computational efficiency can be significantly improved compared to a full matrix multiplication, especially when the matrix is large and sparse. Let's analyze the speedup gained by breaking up a matrix into $b$ blocks when performing matrix multiplication with a block-diagonal matrix.

Assuming the original matrix is $c \times c$ and is broken up into $b$ blocks along each dimension, resulting in $b^2$ smaller matrices of size $\frac{c}{b} \times \frac{c}{b}$, the speedup can be estimated as follows:

1. **Full Matrix Multiplication Complexity**: For a $c \times c$ matrix multiplication, the complexity is $O(c^3)$.

2. **Block-Diagonal Matrix Multiplication Complexity**: When you multiply a matrix with a block-diagonal matrix, only the nonzero blocks of the diagonal matrix contribute to the multiplication. Assuming the diagonal matrix is sparse with $d$ nonzero blocks, the complexity is $O(d \times \left(\frac{c}{b}\right)^3)$.

The speedup $S$ can be calculated as the ratio of the full matrix multiplication complexity to the block-diagonal matrix multiplication complexity:

$$S = \frac{O(c^3)}{O(d \times \left(\frac{c}{b}\right)^3)}$$

Now, let's consider the case where each block of the block-diagonal matrix corresponds to \(s \times s\) non-zero elements:

$$d = b^2 \times s^2$$

Substituting this value into the speedup equation:

$$S = \frac{O(c^3)}{O(b^2 \times s^2 \times \left(\frac{c}{b}\right)^3)}$$

Simplifying:

$$S = \frac{c^3}{b^2 \times s^2 \times \left(\frac{c}{b}\right)^3} = \frac{c^3}{b^3 \times s^2}$$

This means that breaking up the matrix into \(b\) blocks can provide a speedup of $\frac{c^3}{b^3 \times s^2}$when multiplying with a block-diagonal matrix. The actual speedup will depend on the size of the matrix ($c$), the number of blocks ($b$), and the size of the non-zero blocks ($s$).

Keep in mind that this analysis assumes that the matrix multiplication dominates the overall computation, and other factors such as memory access patterns and hardware-specific optimizations can also impact the observed speedup.

## 7.2 What is the downside of having $b$ blocks? How could you fix it, at least partly?

While breaking a matrix into \(b\) blocks for multiplication with a block-diagonal matrix can lead to computational efficiency gains, there are potential downsides associated with having a large number of blocks:

1. **Increased Memory Overhead**: Dividing a matrix into more blocks requires storing more intermediate results and indices, which can increase memory overhead. Each block might have its own memory allocation, leading to more memory fragmentation.

2. **Communication Overhead**: If the multiplication involves communication between blocks (e.g., in a distributed computing environment), the overhead from data movement between different memory regions can impact performance.

3. **Parallel Overhead**: When using parallel processing, managing a large number of blocks can introduce synchronization and management overhead, potentially reducing the efficiency of parallelism.

To mitigate these downsides and maintain some level of computational efficiency, you can consider the following approaches:

1. **Block Size**: Instead of increasing the number of blocks (\(b\)), you can increase the size of each block. This can help reduce memory overhead and improve parallel efficiency. However, increasing block size might lead to less data reuse and cache efficiency, so it's a trade-off.

2. **Optimized Memory Usage**: Utilize memory optimization techniques to minimize memory overhead. This could involve reusing memory buffers, optimizing data layout, and using memory pools to reduce memory fragmentation.

3. **Communication Minimization**: If communication overhead is a concern, consider techniques to minimize data movement between blocks. This might involve algorithms that reduce inter-block dependencies or strategies to overlap computation and communication.

4. **Parallelization Strategy**: If you encounter parallel overhead, explore different parallelization strategies that are suitable for the number of blocks and the hardware you're using. Task-based parallelism, asynchronous execution, or work-stealing can be considered.

5. **Hardware Considerations**: Keep in mind the architecture you're working with. Some hardware accelerators (such as GPUs) have optimizations for specific block sizes, memory access patterns, and parallelization strategies.

In summary, the downside of having a large number of blocks is increased memory, communication, and parallelization overhead. Balancing block size, memory optimization, communication minimization, and parallelization strategies can help mitigate these downsides and maintain computational efficiency.
