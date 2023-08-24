# 1. What happens if you specify the input dimensions to the first layer but not to subsequent layers? Do you get immediate initialization?


```python
import sys
import torch.nn as nn
import torch
import warnings
sys.path.append('/home/jovyan/work/d2l_solutions/notebooks/exercises/d2l_utils/')
import d2l
warnings.filterwarnings("ignore")

net = nn.Sequential(nn.Linear(28*28,256), nn.ReLU(), nn.LazyLinear(10))
net[0].weight
```




    Parameter containing:
    tensor([[-0.0042, -0.0041,  0.0132,  ...,  0.0145, -0.0068, -0.0133],
            [-0.0151,  0.0175,  0.0106,  ...,  0.0133, -0.0010,  0.0254],
            [-0.0012,  0.0225,  0.0020,  ..., -0.0270,  0.0126, -0.0066],
            ...,
            [-0.0037,  0.0327,  0.0219,  ..., -0.0305,  0.0265, -0.0318],
            [-0.0354,  0.0245,  0.0181,  ...,  0.0314, -0.0357, -0.0207],
            [-0.0222,  0.0320,  0.0309,  ...,  0.0284,  0.0331,  0.0284]],
           requires_grad=True)



# 2. What happens if you specify mismatching dimensions?


```python
net = nn.Sequential(nn.Linear(28*28,256), nn.ReLU(), nn.Linear(128,10))
x = torch.randn(1,28*28)
net(x)
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Cell In[7], line 3
          1 net = nn.Sequential(nn.Linear(28*28,256), nn.ReLU(), nn.Linear(128,10))
          2 x = torch.randn(1,28*28)
    ----> 3 net(x)


    File ~/.local/lib/python3.11/site-packages/torch/nn/modules/module.py:1501, in Module._call_impl(self, *args, **kwargs)
       1496 # If we don't have any hooks, we want to skip the rest of the logic in
       1497 # this function, and just call forward.
       1498 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
       1499         or _global_backward_pre_hooks or _global_backward_hooks
       1500         or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1501     return forward_call(*args, **kwargs)
       1502 # Do not call functions when jit is used
       1503 full_backward_hooks, non_full_backward_hooks = [], []


    File ~/.local/lib/python3.11/site-packages/torch/nn/modules/container.py:217, in Sequential.forward(self, input)
        215 def forward(self, input):
        216     for module in self:
    --> 217         input = module(input)
        218     return input


    File ~/.local/lib/python3.11/site-packages/torch/nn/modules/module.py:1501, in Module._call_impl(self, *args, **kwargs)
       1496 # If we don't have any hooks, we want to skip the rest of the logic in
       1497 # this function, and just call forward.
       1498 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
       1499         or _global_backward_pre_hooks or _global_backward_hooks
       1500         or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1501     return forward_call(*args, **kwargs)
       1502 # Do not call functions when jit is used
       1503 full_backward_hooks, non_full_backward_hooks = [], []


    File ~/.local/lib/python3.11/site-packages/torch/nn/modules/linear.py:114, in Linear.forward(self, input)
        113 def forward(self, input: Tensor) -> Tensor:
    --> 114     return F.linear(input, self.weight, self.bias)


    RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x256 and 128x10)


# 3. What would you need to do if you have input of varying dimensionality? Hint: look at the parameter tying.




```python
class TyingParameterMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr, dropouts, k):
        super().__init__()
        self.save_hyperparameters()
        layers = [] # nn.Flatten()
        self.flat = nn.Flatten()
        self.shared = nn.LazyLinear(num_hiddens[0])
        for i in range(1,len(num_hiddens)):
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropouts[i]))
        layers.append(nn.LazyLinear(num_outputs))
        self.net = nn.Sequential(*layers)
        
    def forward(self, X):
        X = self.flat(X)
        r = X.shape[-1] % self.k
        if r != 0:
            pad = torch.zeros(list(X.shape[:-1])+[r])
            X = torch.cat((X, pad), dim=-1)
        n = X.shape[-1] // self.k
        chunks = torch.chunk(X, n, dim=-1)
        tying_X = self.shared(chunks[0])
        for i in range(1,len(chunks)):
            tying_X += self.shared(chunks[i])
        return self.net(tying_X)
```


```python
hparams = {'num_outputs':10,'num_hiddens':[8,4,2],
           'dropouts':[0]*3,'lr':0.1,'k':16}
model = TyingParameterMLP(**hparams)
x1 = torch.randn(1,28*28)
x2 = torch.randn(1,32*32)
print(model(x1))
print(model(x2))
```

    tensor([[ 1.2834, -0.6606, -1.0245,  0.9141, -0.8389, -0.3940, -0.7341, -1.2385,
              0.5124,  1.2274]], grad_fn=<AddmmBackward0>)
    tensor([[ 5.5602, -4.5217, -6.7503,  4.9315, -4.9578, -5.1872, -5.0341, -8.8645,
              0.9688,  6.2508]], grad_fn=<AddmmBackward0>)

