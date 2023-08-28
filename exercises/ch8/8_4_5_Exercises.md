```python
import sys
import torch.nn as nn
import torch
import warnings
sys.path.append('/home/jovyan/work/d2l_solutions/notebooks/exercises/d2l_utils/')
import d2l
from torchsummary import summary
warnings.filterwarnings("ignore")

class Inception(nn.Module):
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super().__init__(*kwargs)
        self.b1 = nn.Sequential(nn.LazyConv2d(c1, kernel_size=1),
                                nn.ReLU())
        self.b2 = nn.Sequential(nn.LazyConv2d(c2[0], kernel_size=1),
                                nn.ReLU(),
                                nn.LazyConv2d(c2[1], kernel_size=3, padding=1),
                                nn.ReLU())
        self.b3 = nn.Sequential(nn.LazyConv2d(c3[0], kernel_size=1),
                                nn.ReLU(),
                                nn.LazyConv2d(c3[1], kernel_size=5, padding=2),
                                nn.ReLU())
        self.b4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                nn.LazyConv2d(c4, kernel_size=1),
                                nn.ReLU())
    
    def forward(self, x):
        o1 = self.b1(x)
        o2 = self.b2(x)
        o3 = self.b3(x)
        o4 = self.b4(x)
        return torch.cat((o1,o2,o3,o4),dim=1)
    
class GoogleNet(d2l.Classifier):
    def b1(self):
        return nn.Sequential(nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def b2(self):
        return nn.Sequential(nn.LazyConv2d(64, kernel_size=1), nn.ReLU(),
                             nn.LazyConv2d(192, kernel_size=3, padding=1),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def b3(self):
        return nn.Sequential(Inception(64, (96, 128), (16, 32), 32),
                             Inception(128, (128, 192), (32, 96), 64),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def b4(self):
        return nn.Sequential(Inception(192, (96, 208), (16, 48), 64),
                             Inception(160, (112, 224), (24, 64), 64),
                             Inception(128, (128, 256), (24, 64), 64),
                             Inception(112, (144, 288), (32, 64), 64),
                             Inception(256, (160, 320), (32, 128), 128),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def b5(self):
        return nn.Sequential(Inception(256, (160, 320), (32, 128), 128),
                             Inception(384, (192, 384), (48, 128), 128),
                             nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
    
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(self.b1(), self.b2(), self.b3(), self.b4(),
                                 self.b5(), nn.LazyLinear(num_classes))
        self.net.apply(d2l.init_cnn)
```

# 1. GoogLeNet was so successful that it went through a number of iterations, progressively improving speed and accuracy. Try to implement and run some of them. They include the following:

## 1.1 Add a batch normalization layer (Ioffe and Szegedy, 2015), as described later in Section 8.5.


```python
class NormInception(nn.Module):
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super().__init__(*kwargs)
        self.b1 = nn.Sequential(nn.LazyConv2d(c1, kernel_size=1), 
                                nn.LazyBatchNorm2d(),
                                nn.ReLU())
        self.b2 = nn.Sequential(nn.LazyConv2d(c2[0], kernel_size=1),
                                nn.LazyBatchNorm2d(),
                                nn.ReLU(),
                                nn.LazyConv2d(c2[1], kernel_size=3, padding=1),
                                nn.LazyBatchNorm2d(),
                                nn.ReLU())
        self.b3 = nn.Sequential(nn.LazyConv2d(c3[0], kernel_size=1),
                                nn.LazyBatchNorm2d(),
                                nn.ReLU(),
                                nn.LazyConv2d(c3[1], kernel_size=5, padding=2),
                                nn.LazyBatchNorm2d(),
                                nn.ReLU())
        self.b4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                nn.LazyBatchNorm2d(),
                                nn.LazyConv2d(c4, kernel_size=1),
                                nn.LazyBatchNorm2d(),
                                nn.ReLU())
    
    def forward(self, x):
        o1 = self.b1(x)
        o2 = self.b2(x)
        o3 = self.b3(x)
        o4 = self.b4(x)
        return torch.cat((o1,o2,o3,o4),dim=1)
    
class NormGoogleNet(d2l.Classifier):
    def b1(self):
        return nn.Sequential(nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
                             nn.LazyBatchNorm2d(),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def b2(self):
        return nn.Sequential(nn.LazyConv2d(64, kernel_size=1),
                             nn.LazyBatchNorm2d(),nn.ReLU(),
                             nn.LazyConv2d(192, kernel_size=3, padding=1),
                             nn.LazyBatchNorm2d(), nn.ReLU(),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def b3(self):
        return nn.Sequential(NormInception(64, (96, 128), (16, 32), 32),
                             NormInception(128, (128, 192), (32, 96), 64),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def b4(self):
        return nn.Sequential(NormInception(192, (96, 208), (16, 48), 64),
                             NormInception(160, (112, 224), (24, 64), 64),
                             NormInception(128, (128, 256), (24, 64), 64),
                             NormInception(112, (144, 288), (32, 64), 64),
                             NormInception(256, (160, 320), (32, 128), 128),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def b5(self):
        return nn.Sequential(NormInception(256, (160, 320), (32, 128), 128),
                             NormInception(384, (192, 384), (48, 128), 128),
                             nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
    
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(self.b1(), self.b2(), self.b3(), self.b4(),
                                 self.b5(), nn.LazyLinear(num_classes))
        self.net.apply(d2l.init_cnn)
```

## 1.2 Make adjustments to the Inception block (width, choice and order of convolutions), as described in Szegedy et al. (2016).


```python
class Inception(nn.Module):
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super().__init__(*kwargs)
        self.b1 = nn.Sequential(nn.LazyConv2d(c1, kernel_size=1),
                                nn.ReLU())
        self.b2 = nn.Sequential(nn.LazyConv2d(c2[0], kernel_size=1),
                                nn.ReLU(),
                                nn.LazyConv2d(c2[1], kernel_size=3, padding=1),
                                nn.ReLU())
        self.b3 = nn.Sequential(nn.LazyConv2d(c3[0], kernel_size=1),
                                nn.ReLU(),
                                nn.LazyConv2d(c3[1], kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.LazyConv2d(c3[2], kernel_size=3, padding=1),
                                nn.ReLU())
        self.b4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                nn.LazyConv2d(c4, kernel_size=1),
                                nn.ReLU())
    
    def forward(self, x):
        o1 = self.b1(x)
        o2 = self.b2(x)
        o3 = self.b3(x)
        o4 = self.b4(x)
        return torch.cat((o1,o2,o3,o4),dim=1)
```


```python
class Inception(nn.Module):
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super().__init__(*kwargs)
        self.b1 = nn.Sequential(nn.LazyConv2d(c1, kernel_size=1),
                                nn.ReLU())
        self.b2 = nn.Sequential(nn.LazyConv2d(c2[0], kernel_size=1),
                                nn.ReLU(),
                                nn.LazyConv2d(c2[1], kernel_size=(1,3), padding=(0,1)),
                                nn.ReLU(),
                                nn.LazyConv2d(c2[2], kernel_size=(3,1), padding=(1,0)),
                                nn.ReLU())
        self.b3 = nn.Sequential(nn.LazyConv2d(c3[0], kernel_size=1),
                                nn.ReLU(),
                                nn.LazyConv2d(c3[1], kernel_size=(1,3), padding=(0,1)),
                                nn.ReLU(),
                                nn.LazyConv2d(c3[2], kernel_size=(3,1), padding=(1,0)),
                                nn.ReLU(),
                                nn.LazyConv2d(c3[3], kernel_size=(1,3), padding=(0,1)),
                                nn.ReLU(),
                                nn.LazyConv2d(c3[4], kernel_size=(3,1), padding=(1,0)),
                                nn.ReLU())
        self.b4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                nn.LazyConv2d(c4, kernel_size=1),
                                nn.ReLU())
    
    def forward(self, x):
        o1 = self.b1(x)
        o2 = self.b2(x)
        o3 = self.b3(x)
        o4 = self.b4(x)
        return torch.cat((o1,o2,o3,o4),dim=1)
```


```python
class Inception(nn.Module):
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super().__init__(*kwargs)
        self.b1 = nn.Sequential(nn.LazyConv2d(c1, kernel_size=1),
                                nn.ReLU())
        self.b2 = nn.Sequential(nn.LazyConv2d(c2[0], kernel_size=1),
                                nn.ReLU())
        self.b2_1 = nn.Sequential(nn.LazyConv2d(c2[1], kernel_size=(1,3), padding=(0,1)),
                                nn.ReLU())
        self.b2_2 = nn.Sequential(nn.LazyConv2d(c2[2], kernel_size=(3,1), padding=(1,0)),
                                nn.ReLU())
        self.b3 = nn.Sequential(nn.LazyConv2d(c3[0], kernel_size=1),
                                nn.ReLU(),
                                nn.LazyConv2d(c3[1], kernel_size=3, padding=1),
                                nn.ReLU())
        self.b3_1 = nn.Sequential(nn.LazyConv2d(c3[2], kernel_size=(1,3), padding=(0,1)),
                                nn.ReLU())
        self.b3_2 = nn.Sequential(nn.LazyConv2d(c3[3], kernel_size=(3,1), padding=(1,0)),
                                nn.ReLU())
        self.b4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                nn.LazyConv2d(c4, kernel_size=1),
                                nn.ReLU())
    
    def forward(self, x):
        o1 = self.b1(x)
        o2 = self.b2(x)
        o2_1 = self.b2_1(o2)
        o2_2 = self.b2_2(o2)
        o3 = self.b3(x)
        o3_1 = self.b3_1(o3)
        o3_2 = self.b3_2(o2)
        o4 = self.b4(x)
        return torch.cat((o1,o2_1,o2_2,o3_1,o3_2,o4),dim=1)
```

## 1.3 Use label smoothing for model regularization, as described in Szegedy et al. (2016).


```python
class LSRGoogleNet(GoogleNet):
    def __init__(self, eps=0, lr=0.1, num_classes=10):
        super().__init__(lr=lr, num_classes=num_classes)
        self.save_hyperparameters()
    
    def loss(self, y_hat, y, averaged=True):
        y_hat = y_hat.reshape((-1, y_hat.shape[-1]))
        y = y.reshape((-1,))
        u = torch.ones(y.shape).tye(torch.float32)/y.shape[-1]
        lsr_loss = (1-self.eps)*F.cross_entropy(y_hat, y, reduction='mean' if averaged else 'none')
        +self.eps*F.cross_entropy(y_hat, u, reduction='mean' if averaged else 'none')
        return lsr_loss
```

## 1.4 Make further adjustments to the Inception block by adding residual connection (Szegedy et al., 2017), as described later in Section 8.6.


```python
class ResInception(nn.Module):
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super().__init__(*kwargs)
        self.b1 = nn.Sequential(nn.LazyConv2d(c1, kernel_size=1), 
                                nn.LazyBatchNorm2d(),
                                nn.ReLU())
        self.b2 = nn.Sequential(nn.LazyConv2d(c2[0], kernel_size=1),
                                nn.LazyBatchNorm2d(),
                                nn.ReLU(),
                                nn.LazyConv2d(c2[1], kernel_size=3, padding=1),
                                nn.LazyBatchNorm2d(),
                                nn.ReLU())
        self.b3 = nn.Sequential(nn.LazyConv2d(c3[0], kernel_size=1),
                                nn.LazyBatchNorm2d(),
                                nn.ReLU(),
                                nn.LazyConv2d(c3[1], kernel_size=5, padding=2),
                                nn.LazyBatchNorm2d(),
                                nn.ReLU())
        self.b4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                nn.LazyBatchNorm2d(),
                                nn.LazyConv2d(c4, kernel_size=1),
                                nn.LazyBatchNorm2d(),
                                nn.ReLU())
    
    def forward(self, x):
        o1 = self.b1(x)+x
        o2 = self.b2(x)+x
        o3 = self.b3(x)+x
        o4 = self.b4(x)+x
        return torch.cat((o1,o2,o3,o4),dim=1)
    
class ResGoogleNet(d2l.Classifier):
    def b1(self):
        return nn.Sequential(nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
                             nn.LazyBatchNorm2d(),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def b2(self):
        return nn.Sequential(nn.LazyConv2d(64, kernel_size=1),
                             nn.LazyBatchNorm2d(),nn.ReLU(),
                             nn.LazyConv2d(192, kernel_size=3, padding=1),
                             nn.LazyBatchNorm2d(), nn.ReLU(),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def b3(self):
        return nn.Sequential(ResInception(64, (96, 128), (16, 32), 32),
                             ResInception(128, (128, 192), (32, 96), 64),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def b4(self):
        return nn.Sequential(ResInception(192, (96, 208), (16, 48), 64),
                             ResInception(160, (112, 224), (24, 64), 64),
                             ResInception(128, (128, 256), (24, 64), 64),
                             ResInception(112, (144, 288), (32, 64), 64),
                             ResInception(256, (160, 320), (32, 128), 128),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    def b5(self):
        return nn.Sequential(ResInception(256, (160, 320), (32, 128), 128),
                             ResInception(384, (192, 384), (48, 128), 128),
                             nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
    
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(self.b1(), self.b2(), self.b3(), self.b4(),
                                 self.b5(), nn.LazyLinear(num_classes))
        self.net.apply(d2l.init_cnn)
```

# 2. What is the minimum image size needed for GoogLeNet to work?

As GoogleNet halve the image 5 times, so the mininum image size needed is $2^5=32$


```python
model = GoogleNet(lr=0.01)
X = torch.randn(1,1,32,32)
model(X)
for m in model.net:
    X = m(X)
    print(X.shape)
```

    torch.Size([1, 64, 8, 8])
    torch.Size([1, 192, 4, 4])
    torch.Size([1, 480, 2, 2])
    torch.Size([1, 832, 1, 1])
    torch.Size([1, 1024])
    torch.Size([1, 10])



```python
model = GoogleNet(lr=0.01)
X = torch.randn(1,1,64,64)
model(X)
for m in model.net:
    X = m(X)
    print(X.shape)
```

    torch.Size([1, 64, 16, 16])
    torch.Size([1, 192, 8, 8])
    torch.Size([1, 480, 4, 4])
    torch.Size([1, 832, 2, 2])
    torch.Size([1, 1024])
    torch.Size([1, 10])


# 3. Can you design a variant of GoogLeNet that works on Fashion-MNIST’s native resolution of $28\times28$ pixels? How would you need to change the stem, the body, and the head of the network, if anything at all?


```python
class SmallGoogLeNet():
    def b1(self):
        return nn.Sequential(nn.LazyConv2d(64, kernel_size=5, stride=1, padding=2),
                             nn.ReLU(),
                             # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                            )

     def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(self.b1(),self.b2(), self.b3(), self.b4(),
                                 self.b5(), nn.LazyLinear(num_classes))
        self.net.apply(d2l.init_cnn)
```

# 4. Compare the model parameter sizes of AlexNet, VGG, NiN, and GoogLeNet. How do the latter two network architectures significantly reduce the model parameter size?

The significant parameter reduction in NiN and GoogLeNet is achieved through the use of 1x1 convolutions and the inception module, respectively. These techniques allow the models to capture features efficiently while keeping the parameter count manageable. The 1x1 convolutions in both NiN and GoogLeNet act as bottleneck layers that help reduce the dimensionality of feature maps, thus leading to fewer parameters in subsequent layers.

|        | GoogLeNet | NiN     | vgg       | alexnet  |
|--------|-----------|---------|-----------|----------|
| params | 5983802   | 2015398 | 128807306 | 46787978 |



```python
model = GoogleNet(lr=0.01)
X = torch.randn(1,3, 224, 224)
_ = model(X)
total_params = sum(p.numel() for p in model.parameters())
print("Total parameters:", total_params)
```

    Total parameters: 5983802


# 5. Compare the amount of computation needed in GoogLeNet and AlexNet. How does this affect the design of an accelerator chip, e.g., in terms of memory size, memory bandwidth, cache size, the amount of computation, and the benefit of specialized operations?

Both GoogLeNet and AlexNet are deep neural network architectures, but they have significant differences in terms of computation requirements due to their architectural designs. The comparison of computation needs between GoogLeNet and AlexNet can have implications for the design of accelerator chips.

1. **Computation Requirements**:
   - **GoogLeNet**: GoogLeNet uses the inception module, which involves multiple parallel convolutional paths and pooling operations. This design introduces a higher level of parallelism, leading to reduced computation within each individual path. Additionally, the use of 1x1 convolutions helps reduce the number of parameters and computations.
   - **AlexNet**: AlexNet has a simpler architecture compared to GoogLeNet. It primarily consists of convolutional and pooling layers, with fewer parallel paths. The overall computation tends to be higher due to the uniform filter sizes and deeper architecture.

2. **Memory Size and Bandwidth**:
   - **GoogLeNet**: The parallel paths in the inception module allow for more efficient memory utilization since each path's intermediate results can be stored separately. This reduces the demand for a large global memory space and might allow for better memory bandwidth utilization.
   - **AlexNet**: The uniform filter sizes and deeper architecture may require larger memory space to store intermediate results, leading to potentially higher memory bandwidth demands.

3. **Cache Size**:
   - **GoogLeNet**: The parallelism in GoogLeNet might benefit from a larger cache size, as different paths can make better use of cache space.
   - **AlexNet**: The deeper architecture and potentially higher temporal locality due to the sequential nature of computation might benefit from a larger cache as well.

4. **Amount of Computation**:
   - **GoogLeNet**: Due to the efficient use of parallelism and dimensionality reduction, GoogLeNet generally requires less computation compared to its accuracy level.
   - **AlexNet**: AlexNet requires more computation due to its uniform filter sizes and deeper architecture.

5. **Specialized Operations**:
   - Both architectures might benefit from specialized operations provided by an accelerator chip. For instance, 1x1 convolutions and depth-wise separable convolutions, which are used in GoogLeNet, can be implemented as hardware-friendly operations in an accelerator chip to further improve efficiency.
   
In terms of designing an accelerator chip for these architectures:

- **Memory**: An accelerator chip for GoogLeNet might require memory structures that can handle the efficient storage and retrieval of intermediate results from parallel paths. For AlexNet, a larger memory capacity might be necessary to accommodate the deeper architecture.

- **Memory Bandwidth**: The chip should ensure sufficient memory bandwidth to facilitate efficient data movement between different layers and paths, taking into consideration the architecture's specific memory access patterns.

- **Cache**: A larger cache size might be beneficial for both architectures to reduce memory access latency and improve temporal locality.

- **Specialized Operations**: The accelerator chip could include specialized hardware units for performing 1x1 convolutions and depth-wise separable convolutions, as these operations are common in modern architectures like GoogLeNet and contribute to performance.

- **Parallelism**: The chip could be designed to exploit parallelism efficiently, especially for architectures like GoogLeNet that utilize parallel paths extensively.

In summary, the computation differences between GoogLeNet and AlexNet impact the design of an accelerator chip in terms of memory, memory bandwidth, cache size, and specialized operations. Understanding the architectural characteristics of these models helps in designing hardware that maximizes efficiency and performance.
