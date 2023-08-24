# 1. Design a layer that takes an input and computes a tensor reduction, i.e., it returns $y_k=\sum_{i,j}W_{i,j,k}x_ix_j$.


```python
import torch.nn as nn
import torch

class ReductionLayer(nn.Module):
    def __init__(self, num_inputs, k):
        super().__init__()
        self.w = nn.ParameterList([nn.Parameter(torch.randn(num_inputs, num_inputs)) for i in range(k)])
        
    def stat_row(self, X):
        y = []
        for part_w in self.w:
            y.append((part_w*X).sum().reshape(-1,1))
        return torch.cat(y,dim=-1)
        
    def forward(self, X):
        chunks = torch.chunk(X,X.shape[0],dim=0)
        rows = []
        for row in chunks:
            row = row.reshape(1,-1)
            part_x = torch.matmul(row.T,row)
            rows.append(self.stat_row(part_x))
        return torch.cat(rows,dim=0)
```


```python
layer = ReductionLayer(5,2)
x = torch.randn(2,5)
layer(x)
```




    tensor([[-4.3252, -3.5257],
            [-5.1311, -0.4626]], grad_fn=<CatBackward0>)



# 2. Design a layer that returns the leading half of the Fourier coefficients of the data.


```python
import torch
import torch.nn as nn

class FourierCoefficientsLayer(nn.Module):
    def __init__(self, num_coefficients):
        super(FourierCoefficientsLayer, self).__init__()
        self.num_coefficients = num_coefficients
    
    def forward(self, x):
        # Apply Fourier transform along the last dimension (assumed to be time dimension)
        fourier_transform = torch.fft.fft(x)
        
        # Select the leading half of the coefficients
        leading_coefficients = fourier_transform[..., :self.num_coefficients]
        
        return leading_coefficients

# Create Fourier coefficients layer with 5 coefficients
num_coefficients = 5
fourier_layer = FourierCoefficientsLayer(num_coefficients)

# Create example input with time dimension (e.g., audio signal)
input_data = torch.randn(1, 10, 10)  # Batch size 1, 10 time steps, 2 features

# Apply the Fourier coefficients layer
output = fourier_layer(input_data)

print("Input shape:", input_data.shape)
print("Output shape:", output.shape)

```

    Input shape: torch.Size([1, 10, 10])
    Output shape: torch.Size([1, 10, 5])

