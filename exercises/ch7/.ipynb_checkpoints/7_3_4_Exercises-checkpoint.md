# 1. Given the final code example in this section with kernel size (3,5), padding (0,1), and stride (3,4), calculate the output shape to check if it is consistent with the experimental result.

$[x_w-k_w+2*pad_w+s_w]//s_w=(8-3+3)//3=2$

$[x_h-k_h+2*pad_h+s_h]//s_h=(8-5+2+4)//4=2$


```python
import torch
import torch.nn as nn
def comp_conv2d(conv2d, X):
    # (1, 1) indicates that batch size and the number of channels are both 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Strip the first two dimensions: examples and channels
    return Y.reshape(Y.shape[2:])

# 1 row and column is padded on either side, so a total of 2 rows or columns
# are added
conv2d = nn.LazyConv2d(1, kernel_size=(3,5), padding=(0,1), stride=(3,4))
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape
```




    torch.Size([2, 2])



# 2. For audio signals, what does a stride of 2 correspond to?

In the context of audio signals and convolutional operations, a stride of 2 corresponds to how much you move the convolutional kernel across the input signal between each computation of the convolution. More specifically, a stride of 2 means that the kernel will "step" or "skip" every 2 units (samples) along the input signal.

When performing convolution with a larger stride, the resulting output will have a reduced length compared to the input. This can be particularly useful for downsampling or reducing the dimensionality of the data. It effectively "compresses" the information in the signal.

For audio signals, a stride of 2 could be used to:

1. **Downsample the Audio**: If you're dealing with audio data and want to reduce its sampling rate or compress the information, you can apply convolution with a stride of 2. This can help in reducing computational load and capturing only essential features.

2. **Feature Extraction**: Strided convolutions can also be useful in extracting important features from audio signals. The higher stride can capture higher-level characteristics of the audio signal, which might be relevant for certain applications like music genre classification or speech recognition.

3. **Pooling Operations**: Strided convolutions are commonly used in conjunction with pooling layers (like MaxPooling) to further reduce the dimensionality of the data. Pooling helps retain the most important features while discarding less relevant information.

Remember that the choice of stride depends on your specific application and goals. A stride of 2 can lead to more aggressive downsampling or feature extraction, while smaller strides (like 1) retain more information but may be computationally more expensive.

# 3. Implement mirror padding, i.e., padding where the border values are simply mirrored to extend tensors.




```python
conv2d = nn.LazyConv2d(1, kernel_size=(3,5), padding=(0,1), stride=(3,4),padding_mode='reflect')
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).sha
```


```python
nn.LazyConv2d??
```


    [0;31mInit signature:[0m
    [0mnn[0m[0;34m.[0m[0mLazyConv2d[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mout_channels[0m[0;34m:[0m [0mint[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mkernel_size[0m[0;34m:[0m [0mUnion[0m[0;34m[[0m[0mint[0m[0;34m,[0m [0mTuple[0m[0;34m[[0m[0mint[0m[0;34m,[0m [0mint[0m[0;34m][0m[0;34m][0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mstride[0m[0;34m:[0m [0mUnion[0m[0;34m[[0m[0mint[0m[0;34m,[0m [0mTuple[0m[0;34m[[0m[0mint[0m[0;34m,[0m [0mint[0m[0;34m][0m[0;34m][0m [0;34m=[0m [0;36m1[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mpadding[0m[0;34m:[0m [0mUnion[0m[0;34m[[0m[0mint[0m[0;34m,[0m [0mTuple[0m[0;34m[[0m[0mint[0m[0;34m,[0m [0mint[0m[0;34m][0m[0;34m][0m [0;34m=[0m [0;36m0[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mdilation[0m[0;34m:[0m [0mUnion[0m[0;34m[[0m[0mint[0m[0;34m,[0m [0mTuple[0m[0;34m[[0m[0mint[0m[0;34m,[0m [0mint[0m[0;34m][0m[0;34m][0m [0;34m=[0m [0;36m1[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mgroups[0m[0;34m:[0m [0mint[0m [0;34m=[0m [0;36m1[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mbias[0m[0;34m:[0m [0mbool[0m [0;34m=[0m [0;32mTrue[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mpadding_mode[0m[0;34m:[0m [0mstr[0m [0;34m=[0m [0;34m'zeros'[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mdevice[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mdtype[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m [0;34m->[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
    [0;31mSource:[0m        
    [0;32mclass[0m [0mLazyConv2d[0m[0;34m([0m[0m_LazyConvXdMixin[0m[0;34m,[0m [0mConv2d[0m[0;34m)[0m[0;34m:[0m  [0;31m# type: ignore[misc][0m[0;34m[0m
    [0;34m[0m    [0;34mr"""A :class:`torch.nn.Conv2d` module with lazy initialization of[0m
    [0;34m    the ``in_channels`` argument of the :class:`Conv2d` that is inferred from[0m
    [0;34m    the ``input.size(1)``.[0m
    [0;34m    The attributes that will be lazily initialized are `weight` and `bias`.[0m
    [0;34m[0m
    [0;34m    Check the :class:`torch.nn.modules.lazy.LazyModuleMixin` for further documentation[0m
    [0;34m    on lazy modules and their limitations.[0m
    [0;34m[0m
    [0;34m    Args:[0m
    [0;34m        out_channels (int): Number of channels produced by the convolution[0m
    [0;34m        kernel_size (int or tuple): Size of the convolving kernel[0m
    [0;34m        stride (int or tuple, optional): Stride of the convolution. Default: 1[0m
    [0;34m        padding (int or tuple, optional): Zero-padding added to both sides of[0m
    [0;34m            the input. Default: 0[0m
    [0;34m        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,[0m
    [0;34m            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``[0m
    [0;34m        dilation (int or tuple, optional): Spacing between kernel[0m
    [0;34m            elements. Default: 1[0m
    [0;34m        groups (int, optional): Number of blocked connections from input[0m
    [0;34m            channels to output channels. Default: 1[0m
    [0;34m        bias (bool, optional): If ``True``, adds a learnable bias to the[0m
    [0;34m            output. Default: ``True``[0m
    [0;34m[0m
    [0;34m    .. seealso:: :class:`torch.nn.Conv2d` and :class:`torch.nn.modules.lazy.LazyModuleMixin`[0m
    [0;34m    """[0m[0;34m[0m
    [0;34m[0m[0;34m[0m
    [0;34m[0m    [0;31m# super class define this variable as None. "type: ignore[..] is required[0m[0;34m[0m
    [0;34m[0m    [0;31m# since we are redefining the variable.[0m[0;34m[0m
    [0;34m[0m    [0mcls_to_become[0m [0;34m=[0m [0mConv2d[0m  [0;31m# type: ignore[assignment][0m[0;34m[0m
    [0;34m[0m[0;34m[0m
    [0;34m[0m    [0;32mdef[0m [0m__init__[0m[0;34m([0m[0;34m[0m
    [0;34m[0m        [0mself[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m        [0mout_channels[0m[0;34m:[0m [0mint[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m        [0mkernel_size[0m[0;34m:[0m [0m_size_2_t[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m        [0mstride[0m[0;34m:[0m [0m_size_2_t[0m [0;34m=[0m [0;36m1[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m        [0mpadding[0m[0;34m:[0m [0m_size_2_t[0m [0;34m=[0m [0;36m0[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m        [0mdilation[0m[0;34m:[0m [0m_size_2_t[0m [0;34m=[0m [0;36m1[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m        [0mgroups[0m[0;34m:[0m [0mint[0m [0;34m=[0m [0;36m1[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m        [0mbias[0m[0;34m:[0m [0mbool[0m [0;34m=[0m [0;32mTrue[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m        [0mpadding_mode[0m[0;34m:[0m [0mstr[0m [0;34m=[0m [0;34m'zeros'[0m[0;34m,[0m  [0;31m# TODO: refine this type[0m[0;34m[0m
    [0;34m[0m        [0mdevice[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m        [0mdtype[0m[0;34m=[0m[0;32mNone[0m[0;34m[0m
    [0;34m[0m    [0;34m)[0m [0;34m->[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m
    [0;34m[0m        [0mfactory_kwargs[0m [0;34m=[0m [0;34m{[0m[0;34m'device'[0m[0;34m:[0m [0mdevice[0m[0;34m,[0m [0;34m'dtype'[0m[0;34m:[0m [0mdtype[0m[0;34m}[0m[0;34m[0m
    [0;34m[0m        [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__init__[0m[0;34m([0m[0;34m[0m
    [0;34m[0m            [0;36m0[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m            [0;36m0[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m            [0mkernel_size[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m            [0mstride[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m            [0mpadding[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m            [0mdilation[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m            [0mgroups[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m            [0;31m# bias is hardcoded to False to avoid creating tensor[0m[0;34m[0m
    [0;34m[0m            [0;31m# that will soon be overwritten.[0m[0;34m[0m
    [0;34m[0m            [0;32mFalse[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m            [0mpadding_mode[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m            [0;34m**[0m[0mfactory_kwargs[0m[0;34m[0m
    [0;34m[0m        [0;34m)[0m[0;34m[0m
    [0;34m[0m        [0mself[0m[0;34m.[0m[0mweight[0m [0;34m=[0m [0mUninitializedParameter[0m[0;34m([0m[0;34m**[0m[0mfactory_kwargs[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m        [0mself[0m[0;34m.[0m[0mout_channels[0m [0;34m=[0m [0mout_channels[0m[0;34m[0m
    [0;34m[0m        [0;32mif[0m [0mbias[0m[0;34m:[0m[0;34m[0m
    [0;34m[0m            [0mself[0m[0;34m.[0m[0mbias[0m [0;34m=[0m [0mUninitializedParameter[0m[0;34m([0m[0;34m**[0m[0mfactory_kwargs[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m[0;34m[0m
    [0;34m[0m    [0;32mdef[0m [0m_get_num_spatial_dims[0m[0;34m([0m[0mself[0m[0;34m)[0m [0;34m->[0m [0mint[0m[0;34m:[0m[0;34m[0m
    [0;34m[0m        [0;32mreturn[0m [0;36m2[0m[0;34m[0m[0;34m[0m[0m
    [0;31mFile:[0m           ~/.local/lib/python3.11/site-packages/torch/nn/modules/conv.py
    [0;31mType:[0m           type
    [0;31mSubclasses:[0m     



```python
import torch

def mirror_padding(input_tensor, padding_size):
    # Check if padding_size is a tuple or single value
    if isinstance(padding_size, int):
        padding_size = (padding_size, padding_size)
    elif len(padding_size) != 2:
        raise ValueError("padding_size should be a single value or a tuple of two values")

    # Get dimensions of input tensor
    batch_size, num_channels, height, width = input_tensor.size()

    # Pad along height and width dimensions
    padded_tensor = torch.nn.functional.pad(input_tensor,
                                            (padding_size[1], padding_size[1], padding_size[0], padding_size[0]),
                                            mode='reflect')

    return padded_tensor

# Example usage
input_tensor = torch.tensor([[[[1, 2], [3, 4]]]]).type(torch.float32)  # Example 1x1x2x2 input tensor
padding_size = 1  # Padding size

padded_tensor = mirror_padding(input_tensor, padding_size)
print("Original tensor:")
print(input_tensor)
print("Padded tensor:")
print(padded_tensor)

```

    Original tensor:
    tensor([[[[1., 2.],
              [3., 4.]]]])
    Padded tensor:
    tensor([[[[4., 3., 4., 3.],
              [2., 1., 2., 1.],
              [4., 3., 4., 3.],
              [2., 1., 2., 1.]]]])


# 4. What are the computational benefits of a stride larger than 1?





# 5. What might be statistical benefits of a stride larger than 1?





# 6. How would you implement a stride of ? What does it correspond to? When would this be useful?


