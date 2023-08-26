# 1. Implement average pooling through a convolution.


```python
import torch
import torch.nn.functional as F

def average_pooling_conv(input_tensor, pool_size):
    batch_size, channels, height, width = input_tensor.size()
    kernel_size = (pool_size, pool_size)
    stride = (pool_size, pool_size)

    # Define the average pooling kernel
    kernel = torch.ones((channels, 1, pool_size, pool_size)) / (pool_size * pool_size)

    # Apply the convolution operation with average pooling kernel
    output_tensor = F.conv2d(input_tensor, kernel, stride=stride, padding=0, groups=channels)

    return output_tensor

# Example usage
input_tensor = torch.randn(1, 1, 6, 6)  # Batch size of 1, 3 channels, 6x6 input
pool_size = 2
output_tensor = average_pooling_conv(input_tensor, pool_size)
print(input_tensor)
print(output_tensor)
```

    tensor([[[[-0.2548, -0.9004,  1.2047, -0.3124,  0.6214,  0.5526],
              [-0.0407, -0.2220, -0.1048, -0.3434, -0.2535,  2.1091],
              [ 0.7223, -0.4832, -1.2391,  0.2195, -1.2479,  0.7798],
              [-0.7320, -1.7425,  0.1385,  1.4043,  0.1163, -0.2561],
              [-0.5119, -0.8785, -0.7798, -1.1799, -1.0041,  0.3349],
              [-0.4481, -0.0313, -1.8601, -0.4983,  0.5341, -0.3495]]]])
    tensor([[[[-0.3545,  0.1110,  0.7574],
              [-0.5589,  0.1308, -0.1520],
              [-0.4674, -1.0795, -0.1211]]]])


# 2. Prove that max-pooling cannot be implemented through a convolution alone.

Max-pooling cannot be implemented through a convolution alone because max-pooling involves a non-linear operation that selects the maximum value within a pooling window, while convolution is a linear operation that computes a weighted sum of values within a kernel window. The non-linearity of max-pooling is essential for its behavior, and it cannot be achieved solely through linear convolution.

# 3. Max-pooling can be accomplished using ReLU operations, i.e., $ReLU(x)=max(0,x)$.

## 3.1 Express $max(a,b)$ by using only ReLU operations.

$max(a,b) = ReLU(a-b)+b$


```python
import random
def relu(x):
    return max(x,0)
# 生成随机整数
a = random.randint(-100, 100)  # 生成1到100之间的随机整数
b = random.randint(-100, 100)  # 生成1到100之间的随机整数
print(a,b,max(a,b))
relu(a-b)+b == max(a,b)
```

    4 94 94





    True




```python
a = torch.tensor([2,3])
b = torch.tensor([1,5])
c = torch.max(a,b)
(relu(a-b)+b == c).all()
```




    tensor(True)



## 3.2 Use this to implement max-pooling by means of convolutions and ReLU layers.


```python
def max_pooling_conv(input_tensor, pool_size):
    batch_size, channels, height, width = input_tensor.size()
    kernel_size = (pool_size, pool_size)
    stride = (pool_size, pool_size)
    output_shape = F.conv2d(input_tensor, torch.ones((channels, 1, pool_size, pool_size)), stride=stride, padding=0, groups=channels).shape
    output_tensor = torch.tensor([-torch.inf]*output_shape.numel()).reshape(output_shape)
    # Define the average pooling kernel
    for i in range(pool_size*pool_size):
        kernel = torch.zeros(pool_size*pool_size)
        kernel[i] = 1
        kernel = kernel.reshape(channels, 1, pool_size, pool_size)
    # Apply the convolution operation with average pooling kernel
        temp = F.conv2d(input_tensor, kernel, stride=stride, padding=0, groups=channels)
        output_tensor = relu(output_tensor - temp) + temp
    return output_tensor

# Example usage
input_tensor = torch.randn(1, 1, 6, 6)  # Batch size of 1, 3 channels, 6x6 input
pool_size = 2
output_tensor = max_pooling_conv(input_tensor, pool_size)
print(input_tensor)
print(output_tensor)
```

    tensor([[[[-1.3226,  0.2061, -0.1055, -0.0231,  1.1557,  0.0235],
              [ 0.7217, -0.4185,  0.1732,  0.5446,  0.5935, -0.3394],
              [ 0.1075, -0.9181, -0.6921,  0.5263, -1.5948, -1.0824],
              [ 0.2959,  1.0360, -0.8432,  0.0958, -0.2052,  0.8342],
              [ 2.1395,  0.1743, -1.5479, -0.4525,  0.4780, -0.5466],
              [ 0.9333, -1.1564,  1.0396,  0.4543, -0.3832, -1.0517]]]])
    tensor([[[[0.7217, 0.5446, 1.1557],
              [1.0360, 0.5263, 0.8342],
              [2.1395, 1.0396, 0.4780]]]])


## 3.3 How many channels and layers do you need for a $2\times 2$ convolution? How many for a $3\times 3$ convolution?

$n\times n$ convolution needs $n^2$ channels and layer, so we need 4 for $2\times 2$ convolution, 9 for $3\times 3$convolution.

# 4. What is the computational cost of the pooling layer? Assume that the input to the pooling layer is of size $c\times h\times w$, the pooling window has a shape of $p_h\times p_w$ with a padding of $(p_h,p_w)$ and a stride of $(s_h,s_w)$.

$c\times[(h+p_h-p_h+s_h)//s_h]\times[(w+p_w-p_w+s_w)//s_w]\times p_h\times p_w$

# 5. Why do you expect max-pooling and average pooling to work differently?

Max-pooling and average pooling are two different types of pooling operations commonly used in convolutional neural networks (CNNs) for downsampling input feature maps. They work differently due to the nature of the aggregation strategies they employ.

1. **Max-Pooling**:
   - Max-pooling selects the maximum value from the pooling window and retains it as the representative value for that region.
   - Max-pooling is particularly effective at capturing the most prominent features within the window. It emphasizes the presence of strong activations and helps to identify significant patterns in the data.
   - It is robust to noise and variations in the data, as it prioritizes the most dominant information.

2. **Average Pooling**:
   - Average pooling calculates the average value of all the elements within the pooling window and uses this average as the representative value for that region.
   - Average pooling provides a more smoothed representation of the data. It can help capture a broader understanding of the data distribution within the window.
   - It can mitigate the impact of outliers and noise in the data, as extreme values have less influence due to the averaging.

Due to these differences, max-pooling and average pooling tend to work differently in various scenarios:

- **Feature Detection**: Max-pooling is better suited for detecting distinctive features, edges, and patterns in the input. It retains the most salient activations, helping the network focus on significant details.
- **Global Information**: Average pooling captures global information within a region, providing a broader context. This can be useful when a more generalized understanding of the data is needed.
- **Robustness**: Max-pooling is robust to noise and can handle variations well. Average pooling can smooth out variations and mitigate noise to some extent.

In practice, the choice between max-pooling and average pooling depends on the task, the nature of the data, and the architecture of the neural network. In many cases, max-pooling is preferred for its feature-preserving properties and ability to capture distinctive patterns. Average pooling might be useful when a more holistic representation or noise reduction is desired.

# 6. Do we need a separate minimum pooling layer? Can you replace it with another operation?

We can replace minium poolying layer with $-max(-a,-b)$


```python
a = torch.tensor([2,3])
b = torch.tensor([1,5])
(-torch.max(-a,-b) == torch.min(a,b)).all()
```




    tensor(True)



# 7. We could use the softmax operation for pooling. Why might it not be so popular?

Using the softmax operation for pooling is theoretically possible, but it's not a commonly used approach in practice for several reasons:

1. **Numerical Stability**: Softmax is sensitive to the scale of input values. When applied to a large set of values, it involves exponentiation, which can lead to numerical instability and overflow issues. This can cause the gradients during backpropagation to become very small or large, affecting the learning process.

2. **Normalization**: Softmax is a normalization operation that converts input values into probabilities that sum to 1. This normalization means that the pooled values are context-dependent and relative to the other values in the pooling window. This might not always be suitable for pooling, where the goal is to retain the most prominent features, rather than redistributing them as probabilities.

3. **Loss of Discriminative Information**: Softmax pooling could lead to a loss of discriminative information. It doesn't emphasize the strongest activations as effectively as max-pooling does, potentially reducing the network's ability to detect important features.

4. **Complexity**: Softmax pooling introduces additional computational complexity due to the exponentiation and normalization steps. This can slow down training and inference times, making it less efficient compared to other pooling methods.

5. **Lack of Invariance**: Pooling operations like max-pooling and average pooling provide invariance to small translations or perturbations in the input data, making them more robust. Softmax pooling, being context-dependent and sensitive to the distribution of values, might not offer the same level of invariance.

6. **Architectural Consistency**: Softmax is commonly used in the output layer of a neural network for classification tasks. Using it as a pooling operation might break the architectural consistency and clear separation between different layers in the network.

In summary, while softmax pooling is theoretically possible, it's not widely adopted in practice due to its numerical instability, normalization effects, potential loss of discriminative information, increased complexity, lack of invariance, and inconsistency with common architectural choices. Max-pooling and average pooling remain the dominant choices for pooling operations in most convolutional neural network architectures.
