# 1. Construct an image X with diagonal edges.


```python
import torch

def corr2d(X,K):
    h,w = K.shape
    Y = torch.zeros(X.shape[0]-h+1, X.shape[1]-w+1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w]*K).sum()
    return Y

K = torch.tensor([[1.0,-1.0]])
X = torch.eye(6,8)
X
```




    tensor([[1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0.]])



## 1.1 What happens if you apply the kernel K in this section to it?


```python
corr2d(X, K)
```




    tensor([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.],
            [-1.,  1.,  0.,  0.,  0.,  0.,  0.],
            [ 0., -1.,  1.,  0.,  0.,  0.,  0.],
            [ 0.,  0., -1.,  1.,  0.,  0.,  0.],
            [ 0.,  0.,  0., -1.,  1.,  0.,  0.],
            [ 0.,  0.,  0.,  0., -1.,  1.,  0.]])



## 1.2 What happens if you transpose X?


```python
corr2d(X.T, K)
```




    tensor([[ 1.,  0.,  0.,  0.,  0.],
            [-1.,  1.,  0.,  0.,  0.],
            [ 0., -1.,  1.,  0.,  0.],
            [ 0.,  0., -1.,  1.,  0.],
            [ 0.,  0.,  0., -1.,  1.],
            [ 0.,  0.,  0.,  0., -1.],
            [ 0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.]])



## 1.3 What happens if you transpose K?


```python
corr2d(X, K.T)
```




    tensor([[ 1., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  1., -1.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  1., -1.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  1., -1.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  1., -1.,  0.,  0.]])



# 2. Design some kernels manually.

## 2.1 Given a directional vector $\vec{v}=(v_1,v_2)$, derive an edge-detection kernel that detects edges orthogonal to $\vec{v}$, i.e., edges in the direction $(v_2,-v_1)$.


```python
pic = torch.roll(X, shifts=(1,),dims=(1,))+torch.roll(X, shifts=(2,),dims=(1,))+X
Y = X+torch.roll(X, shifts=(2,),dims=(1,))
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = nn.LazyConv2d(1, kernel_size=(3, 3), bias=False,padding=1)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
train_X = pic.reshape((1, 1, pic.shape[0], pic.shape[1]))
train_Y = Y.reshape((1, 1, Y.shape[0], Y.shape[1]))
lr = 1e-2  # Learning rate

for i in range(16):
    Y_hat = conv2d(train_X)
    l = (Y_hat - train_Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # Update the kernel
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')
conv2d.weight.data.squeeze()
```

    epoch 2, loss 10.199
    epoch 4, loss 5.911
    epoch 6, loss 4.149
    epoch 8, loss 3.191
    epoch 10, loss 2.631
    epoch 12, loss 2.293
    epoch 14, loss 2.085
    epoch 16, loss 1.954





    tensor([[ 0.3847, -0.3420,  0.2657],
            [-0.1117,  0.3634, -0.0246],
            [ 0.1967, -0.1634,  0.2337]])




```python
corr2d(pic,conv2d.weight.data.squeeze())
```




    tensor([[ 0.9362,  0.3084,  0.9182, -0.1221,  0.2900,  0.0000],
            [-0.1041,  0.9362,  0.3084,  0.9182, -0.1221,  0.2900],
            [ 0.2157, -0.1041,  0.9362,  0.3084,  0.9182, -0.1221],
            [ 0.0000,  0.2157, -0.1041,  0.9362,  0.3084,  0.9182]])




```python
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tF
def stat_theta(r):
    v = [1,0]
    theta = math.acos(F.cosine_similarity(torch.tensor(r).type(torch.float32),torch.tensor(v).type(torch.float32),dim=0))/math.pi*180
    return theta

def gen_K(v):
    v = torch.tensor(v, dtype=torch.float32)  # Replace v1 and v2 with your values
    u = v / torch.norm(v)
    # Create the edge-detection kernel along the direction (v2, -v1)
    K = torch.tensor([[-u[1], u[0]],[-u[0], -u[1]]], dtype=torch.float32)
    return K

def test(r,a):
    theta = stat_theta(r)
    K = gen_K(r)
    print(K)
    b = tF.rotate(a.reshape(1,1,a.shape[0],-1,),angle=theta).reshape(a.shape[0],-1)
    print(b)
    print(corr2d(b,K))
```


```python
a = torch.ones((6, 8))
a[:, 2:6] = 0
test([1,0],a)
```

    tensor([[-0.,  1.],
            [-1., -0.]])
    tensor([[1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.]])
    tensor([[ 0., -1.,  0.,  0.,  0.,  1.,  0.],
            [ 0., -1.,  0.,  0.,  0.,  1.,  0.],
            [ 0., -1.,  0.,  0.,  0.,  1.,  0.],
            [ 0., -1.,  0.,  0.,  0.,  1.,  0.],
            [ 0., -1.,  0.,  0.,  0.,  1.,  0.]])



```python
test([0,1],a)
```

    tensor([[-1.,  0.],
            [-0., -1.]])
    tensor([[0., 1., 1., 1., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 1., 1., 1., 0.]])
    tensor([[ 0., -1., -1., -1., -1., -1., -1.],
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [-1., -1., -1., -1., -1., -1.,  0.]])



```python
test([1,1],a)
```

    tensor([[-0.7071,  0.7071],
            [-0.7071, -0.7071]])
    tensor([[0., 0., 0., 0., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0., 1., 1., 1.],
            [1., 0., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 0., 1.],
            [1., 1., 1., 0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 0., 0., 0., 0.]])
    tensor([[ 0.0000,  0.0000,  0.0000,  0.7071, -0.7071, -1.4142, -2.1213],
            [-0.7071,  0.0000,  0.0000,  0.0000,  0.7071, -0.7071, -1.4142],
            [-2.1213, -0.7071,  0.0000,  0.0000,  0.0000,  0.7071, -0.7071],
            [-1.4142, -2.1213, -0.7071,  0.0000,  0.0000,  0.0000,  0.7071],
            [-0.7071, -1.4142, -2.1213, -0.7071,  0.0000,  0.0000,  0.0000]])


## 2.2 Derive a finite difference operator for the second derivative. What is the minimum size of the convolutional kernel associated with it? Which structures in images respond most strongly to it?

The second derivative of a continuous function can be approximated using a finite difference operator. One common way to do this is to use the central difference formula, which is given by:

$$ \frac{\partial^2 f}{\partial x^2} \approx \frac{f(x+h) - 2f(x) + f(x-h)}{h^2} $$

Where $h$ is a small step size.

To create a convolutional kernel associated with the second derivative, we can discretize the above formula and put it into a kernel format. The kernel would look like:
$$\text{kernel} = \begin{bmatrix} 1 & -2 & 1 \end{bmatrix}$$
This kernel captures the second derivative along the horizontal direction. It's worth noting that the central difference formula can be applied in both horizontal and vertical directions separately to capture the second derivative along each direction.

The minimum size of the convolutional kernel associated with the second derivative is $3 \times 1$ or $1 \times 3$. This size captures the essence of the central difference formula for the second derivative.

Structures in images that have rapid intensity changes or sharp transitions will respond most strongly to this second derivative kernel. These structures include edges, corners, and other high-frequency features. The second derivative kernel enhances areas in the image where the intensity changes abruptly, making it a useful tool for edge detection and feature extraction.


```python
K = torch.tensor([[1,-2,1]])
print(a)
corr2d(a,K)
```

    tensor([[1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.]])





    tensor([[-1.,  1.,  0.,  0.,  1., -1.],
            [-1.,  1.,  0.,  0.,  1., -1.],
            [-1.,  1.,  0.,  0.,  1., -1.],
            [-1.,  1.,  0.,  0.,  1., -1.],
            [-1.,  1.,  0.,  0.,  1., -1.],
            [-1.,  1.,  0.,  0.,  1., -1.]])



## 2.3 How would you design a blur kernel? Why might you want to use such a kernel?

Designing a blur kernel involves creating a convolutional kernel that, when applied to an image, reduces the high-frequency components in the image, resulting in a smoother and more blurred appearance. A commonly used blur kernel is the Gaussian kernel, which is derived from the Gaussian distribution. The Gaussian kernel has the property of spreading out the pixel values around the central pixel, creating a gradual transition between neighboring pixels.

To design a Gaussian blur kernel, you typically follow these steps:

1. Choose the size of the kernel: The size of the kernel determines the extent of blurring. A larger kernel size will result in more pronounced blurring.

2. Determine the standard deviation (\(\sigma\)): The standard deviation controls the spread of the Gaussian distribution. A larger \(\sigma\) will result in a wider spread and more smoothing.

3. Compute the Gaussian values: For each pixel in the kernel, compute the Gaussian value based on its distance from the center. The Gaussian values are then normalized to ensure that they sum up to 1.

Here's an example of how you can create a 2D Gaussian blur kernel using Python and NumPy:



```python

import numpy as np
def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-size//2)**2 + (y-size//2)**2) / (2*sigma**2)),
        (size, size)
    )
    kernel /= np.sum(kernel)
    return kernel

kernel_size = 3
sigma = 1.0
blur_kernel = torch.tensor(gaussian_kernel(kernel_size, sigma))

print(f"Blur kernel sum:{blur_kernel.sum()}")
print(blur_kernel)
print(a)
corr2d(a,blur_kernel)
```

    Blur kernel sum:1.0
    tensor([[0.0751, 0.1238, 0.0751],
            [0.1238, 0.2042, 0.1238],
            [0.0751, 0.1238, 0.0751]], dtype=torch.float64)
    tensor([[1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.]])





    tensor([[0.7259, 0.2741, 0.0000, 0.0000, 0.2741, 0.7259],
            [0.7259, 0.2741, 0.0000, 0.0000, 0.2741, 0.7259],
            [0.7259, 0.2741, 0.0000, 0.0000, 0.2741, 0.7259],
            [0.7259, 0.2741, 0.0000, 0.0000, 0.2741, 0.7259]])



Why might you want to use a blur kernel?

1. **Noise Reduction**: Blurring can help reduce noise and unwanted artifacts in an image. High-frequency noise is often smoothed out, resulting in a cleaner appearance.

2. **Image Smoothing**: Blurring is commonly used to smooth out textures and fine details in an image, creating a more cohesive and aesthetically pleasing result.

3. **Edge Preservation**: While blurring reduces high-frequency details, certain blur techniques can preserve important edges while still providing a smoother overall appearance.

4. **Preprocessing**: Blurring can be used as a preprocessing step for various computer vision tasks such as object detection and recognition, where the focus is on features rather than fine textures.

5. **Privacy Protection**: Blurring or pixelating specific regions of an image can be used for privacy protection by making sensitive information less recognizable.

6. **Artistic Effects**: Blurring can also be used creatively to achieve artistic effects or simulate depth of field in photography.

Overall, blur kernels serve as a versatile tool in image processing with applications ranging from noise reduction to artistic manipulation.

## 2.4 What is the minimum size of a kernel to obtain a derivative of order $d$?

need a kernel of size $2d+1$ along the direction in which we're calculating the derivative.

One possible way to get the kernel size of k-order derivative in 1D is to use the finite difference approximation, which estimates the derivative of a function at a point by using the values of the function at nearby points. ¹ For example, if we use the central difference formula to approximate the derivative, then we need a kernel of size 2k+1 to obtain a derivative of order k. This is because the central difference formula uses k points on each side of the center point to estimate the derivative. For instance, the first-order derivative can be approximated by using a kernel of size 3: $$\frac{\partial f}{\partial x}(x)\approx \frac{f(x+1)-f(x-1)}{2}$$ The second-order derivative can be approximated by using a kernel of size 5: $$\frac{\partial^2 f}{\partial x^2}(x)\approx \frac{f(x+2)-2f(x)+f(x-2)}{4}$$ And so on. However, if we use other types of kernels, such as Sobel or Laplace kernels, then we may need different sizes to obtain a derivative of order k. For example, the Sobel kernel can approximate the first-order derivative by using a kernel of size 3, but it cannot approximate the second-order derivative by using a single kernel. Instead, we need to apply the Sobel kernel twice or use another kernel, such as the Laplace kernel, which can approximate the second-order derivative by using a kernel of size 3. 

# 3. When you try to automatically find the gradient for the Conv2D class we created, what kind of error message do you see?


```python
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```


```python
x = pic
# Create an instance of your custom Conv2D layer
custom_conv2d = Conv2D(kernel_size=(3, 3))

# Set requires_grad to True
x.requires_grad = True

# Perform forward pass
output = custom_conv2d(x)

# Compute gradients
output.backward()

# Access gradients
print("Gradient of weight:", custom_conv2d.weight.grad)
print("Gradient of bias:", custom_conv2d.bias.grad)
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Cell In[404], line 12
          9 output = custom_conv2d(x)
         11 # Compute gradients
    ---> 12 output.backward()
         14 # Access gradients
         15 print("Gradient of weight:", custom_conv2d.weight.grad)


    File ~/.local/lib/python3.11/site-packages/torch/_tensor.py:487, in Tensor.backward(self, gradient, retain_graph, create_graph, inputs)
        477 if has_torch_function_unary(self):
        478     return handle_torch_function(
        479         Tensor.backward,
        480         (self,),
       (...)
        485         inputs=inputs,
        486     )
    --> 487 torch.autograd.backward(
        488     self, gradient, retain_graph, create_graph, inputs=inputs
        489 )


    File ~/.local/lib/python3.11/site-packages/torch/autograd/__init__.py:193, in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
        189 inputs = (inputs,) if isinstance(inputs, torch.Tensor) else \
        190     tuple(inputs) if inputs is not None else tuple()
        192 grad_tensors_ = _tensor_or_tensors_to_tuple(grad_tensors, len(tensors))
    --> 193 grad_tensors_ = _make_grads(tensors, grad_tensors_, is_grads_batched=False)
        194 if retain_graph is None:
        195     retain_graph = create_graph


    File ~/.local/lib/python3.11/site-packages/torch/autograd/__init__.py:88, in _make_grads(outputs, grads, is_grads_batched)
         86 if out.requires_grad:
         87     if out.numel() != 1:
    ---> 88         raise RuntimeError("grad can be implicitly created only for scalar outputs")
         89     new_grads.append(torch.ones_like(out, memory_format=torch.preserve_format))
         90 else:


    RuntimeError: grad can be implicitly created only for scalar outputs


# 4. How do you represent a cross-correlation operation as a matrix multiplication by changing the input and kernel tensors?


```python
def corr2d_matmul(pic, K):
    pad_K = F.pad(K,(0,pic.shape[1]-K.shape[1],0,pic.shape[0]-K.shape[0])).type(torch.float32)
    l = []
    for i in range(pic.shape[0]-K.shape[0]+1):
        for j in range(pic.shape[1]-K.shape[1]+1):
            l.append(torch.roll(pad_K,(i,j),(0,1)).reshape(1,-1))
    print(torch.cat(l,dim=0))
    return (torch.cat(l,dim=0)@pic.reshape(-1,1)).reshape(pic.shape[0]-K.shape[0]+1,pic.shape[1]-K.shape[1]+1)
```


```python
K = torch.tensor([[1,1],[1,1]])
x = torch.ones(3,3)
(corr2d_matmul(x,K) == corr2d(x,K)).all()
```

    tensor([[1., 1., 0., 1., 1., 0., 0., 0., 0.],
            [0., 1., 1., 0., 1., 1., 0., 0., 0.],
            [0., 0., 0., 1., 1., 0., 1., 1., 0.],
            [0., 0., 0., 0., 1., 1., 0., 1., 1.]])





    tensor(True)


