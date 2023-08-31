# 1. What are the major differences between the Inception block in Fig. 8.4.1 and the residual block? How do they compare in terms of computation, accuracy, and the classes of functions they can describe?



Inception blocks and residual blocks are two distinct architectural components commonly used in deep neural networks for various tasks, including image classification. Let's compare the major differences between the Inception block (also known as GoogLeNet inception module) and the residual block (used in ResNet architectures):

**Inception Block (GoogLeNet Inception Module)**:
The Inception block, introduced in the GoogLeNet architecture, is designed to capture features at multiple scales by using parallel convolutional layers of different sizes and pooling operations. It aims to create a rich hierarchy of features by combining information from different receptive fields. The major characteristics of the Inception block are:

1. **Parallel Convolutions**: The Inception block contains multiple convolutional layers of different kernel sizes (e.g., 1x1, 3x3, 5x5). These parallel convolutions capture features of different scales and capture both fine and coarse details.

2. **Pooling Operations**: The Inception block also includes pooling operations, such as max-pooling, which helps reduce spatial dimensions and capture translational invariance.

3. **Concatenation**: The outputs of the parallel convolutions and pooling operations are concatenated along the channel dimension. This allows the network to capture a diverse set of features.

**Residual Block**:
The residual block, introduced in the ResNet architecture, is designed to address the vanishing gradient problem and enable the training of very deep networks. It introduces skip connections (also known as shortcut connections) that pass the input of a layer directly to a later layer. The major characteristics of the residual block are:

1. **Skip Connections**: The residual block uses a skip connection that adds the original input (identity) to the output of the convolutional layers. This creates a "residual" or a "shortcut" path for information to flow directly through the network.

2. **Identity Mapping**: The idea behind the residual block is that the model can learn to adjust the weights of the convolutional layers to make them represent the residual (the difference between the input and the output). This helps mitigate vanishing gradient issues.

3. **Batch Normalization**: Residual blocks often include batch normalization after each convolutional layer. This helps stabilize and accelerate training.

4. **Two-Path Learning**: The residual block essentially learns a residual transformation, which can be viewed as a combination of "what to add" (learned by convolutional layers) and "what to keep" (passed through the skip connection).

In summary, the major difference between the Inception block and the residual block lies in their architectural goals and design principles. The Inception block focuses on capturing features at different scales using parallel operations, while the residual block focuses on enabling the training of very deep networks by introducing skip connections that facilitate the learning of residual transformations. Both architectural components have been instrumental in advancing the capabilities of deep neural networks for various tasks.

# 2. Refer to Table 1 in the ResNet paper (He et al., 2016) to implement different variants of the network.




```python
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
sys.path.append('/home/jovyan/work/d2l_solutions/notebooks/exercises/d2l_utils/')
import d2l

class Residual(nn.Module):
    def __init__(self, convs, conv_1x1_channel, strides=1):
        super().__init__()
        layers = []
        for i,conv in enumerate(convs):
            num_channels, kernel_size, padding = conv
            conv_strides = 1 if i != 0 else strides
            layers.append(nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=conv_strides))
            layers.append(nn.LazyBatchNorm2d())
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers[:-1])
        self.conv = None
        if conv_1x1_channel:
            self.conv = nn.LazyConv2d(conv_1x1_channel, kernel_size=1, stride=strides)
        
        
    def forward(self, X):
        Y = self.net(X)
        if self.conv:
            X = self.conv(X)
        Y += X
        return F.relu(Y)
        
class ResNet(d2l.Classifier):
    def block(self, num_residuals, convs, conv_1x1_channel, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(convs, conv_1x1_channel,strides=2))
            else:
                blk.append(Residual(convs, conv_1x1_channel))
        return nn.Sequential(*blk)
    
    def __init__(self, arch, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        for i, b in enumerate(arch):
            self.net.add_module(f'b{i+2}', self.block(*b, first_block=(i==0)))
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(d2l.init_cnn)
        
def experiment(data, model):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
    trainer.fit(model, data)
    X,y = next(iter(data.get_dataloader(False)))
    X = X.to('cuda')
    y = y.to('cuda')
    y_hat = model(X)
    return model.accuracy(y_hat,y).item()
```


```python
data = d2l.FashionMNIST(batch_size=64, resize=(224, 224))
arch18 = [(2,[(64,3,1)]*2,None),(2,[(128,3,1)]*2,None),(2,[(256,3,1)]*2,None),(2,[(512,3,1)]*2,None)]
resnet18 = ResNet(arch=arch18, lr=0.01)
experiment(data, resnet18)
```


```python
arch34 = [(3,[(64,3,1)]*2,None),(4,[(128,3,1)]*2,None),(6,[(256,3,1)]*2,None),(3,[(512,3,1)]*2,None)]
resnet34 = ResNet(arch=arch34, lr=0.01)
experiment(data, resnet34)
```

# 3. For deeper networks, ResNet introduces a “bottleneck” architecture to reduce model complexity. Try to implement it.




```python
arch50 = [(3,[(64,1,0),(64,3,1)],256),(4,[(128,1,0),(128,3,1)],512),(6,[(256,1,0),(256,3,1)],1024),(3,[(512,1,0),(512,3,1)],2048)]
resnet50 = ResNet(arch=arch50, lr=0.01)
experiment(data, resnet50)
```


```python
arch101 = [(3,[(64,1,0),(64,3,1)],256),(4,[(128,1,0),(128,3,1)],512),(23,[(256,1,0),(256,3,1)],1024),(3,[(512,1,0),(512,3,1)],2048)]
resnet101 = ResNet(arch=arch101, lr=0.01)
experiment(data, resnet101)
```


```python
arch152 = [(3,[(64,1,0),(64,3,1)],256),(8,[(128,1,0),(128,3,1)],512),(36,[(256,1,0),(256,3,1)],1024),(3,[(512,1,0),(512,3,1)],2048)]
resnet152 = ResNet(arch=arch152, lr=0.01)
experiment(data, resnet152)
```

# 4. In subsequent versions of ResNet, the authors changed the “convolution, batch normalization, and activation” structure to the “batch normalization, activation, and convolution” structure. Make this improvement yourself. See Figure 1 in He et al. (2016) for details.


```python
class SubResidual(nn.Module):
    def __init__(self, convs, conv_1x1_channel, strides=1):
        super().__init__()
        layers = []
        for i,conv in enumerate(convs):
            num_channels, kernel_size, padding = conv
            conv_strides = 1 if i != 0 else strides
            layers.append(nn.LazyBatchNorm2d())
            layers.append(nn.ReLU())
            layers.append(nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=conv_strides))
        self.net = nn.Sequential(*layers[:-1])
        self.conv = None
        if conv_1x1_channel:
            self.conv = nn.LazyConv2d(conv_1x1_channel, kernel_size=1, stride=strides)
        
        
    def forward(self, X):
        Y = self.net(X)
        if self.conv:
            X = self.conv(X)
        Y += X
        return F.relu(Y)
        
class SubResNet(d2l.Classifier):
    def block(self, num_residuals, convs, conv_1x1_channel, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(SubResidual(convs, conv_1x1_channel,strides=2))
            else:
                blk.append(SubResidual(convs, conv_1x1_channel))
        return nn.Sequential(*blk)
    
    def __init__(self, arch, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        for i, b in enumerate(arch):
            self.net.add_module(f'b{i+2}', self.block(*b, first_block=(i==0)))
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(d2l.init_cnn)
```


```python
arch18 = [(2,[(64,3,1)]*2,None),(2,[(128,3,1)]*2,None),(2,[(256,3,1)]*2,None),(2,[(512,3,1)]*2,None)]
resnet18 = SubResNet(arch=arch18, lr=0.01)
experiment(data, resnet18)
```

# 5. Why can’t we just increase the complexity of functions without bound, even if the function classes are nested?

Increasing the complexity of functions without bound, even when using nested function classes, can lead to several issues in machine learning and model training. There are fundamental challenges related to model capacity, overfitting, computational complexity, and generalization. Here's why it's not a practical approach:

1. **Overfitting**: When you increase the complexity of a model without considering the data's true underlying patterns, the model can start fitting noise in the training data. This leads to overfitting, where the model performs very well on the training data but poorly on unseen data.

2. **Computational Complexity**: Complex models with many parameters require more computational resources and time for training and inference. This can lead to practical challenges in terms of training time, memory usage, and scalability.

3. **Diminishing Returns**: Increasing model complexity does not necessarily result in proportionate improvements in performance. There's a point where adding more complexity provides marginal or diminishing returns in terms of improved accuracy.

4. **Generalization**: A model's primary goal is to generalize well to new, unseen data. If the model becomes too complex, it may become overly specialized to the training data and fail to generalize to new instances.

5. **Regularization Challenges**: Without proper regularization techniques, increasing complexity can exacerbate overfitting. Regularization techniques help control model complexity and prevent overfitting.

6. **Interpretability**: Highly complex models can become difficult to interpret, making it hard to understand their decision-making processes and diagnose issues.

7. **Data Efficiency**: Simpler models are often more data-efficient. Extremely complex models might require vast amounts of training data to generalize well.

8. **Bias-Variance Trade-off**: Increasing model complexity influences the balance between bias (underfitting) and variance (overfitting). Finding the right balance is crucial for good performance.

Instead of unbounded complexity, it's more effective to choose model architectures that strike the right balance between capacity and generalization. Techniques like regularization, cross-validation, and ensemble methods can help improve model performance without unnecessarily increasing complexity. Ultimately, the goal is to build models that can capture the underlying patterns in the data while avoiding overfitting and computational inefficiencies.
