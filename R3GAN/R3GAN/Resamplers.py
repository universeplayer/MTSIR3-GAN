import torch
import torch.nn as nn
import numpy
from torch_utils.ops import upfirdn2d

def CreateLowpassKernel(Weights, Inplace):
    Kernel = numpy.array([Weights]) if Inplace else numpy.convolve(Weights, [1, 1]).reshape(1, -1)
    Kernel = torch.Tensor(Kernel.T @ Kernel)
    return Kernel / torch.sum(Kernel)

class InterpolativeUpsamplerReference(nn.Module):
    def __init__(self, Filter):
        super(InterpolativeUpsamplerReference, self).__init__()
        
        self.register_buffer('Kernel', CreateLowpassKernel(Filter, Inplace=False))
        self.FilterRadius = len(Filter) // 2
        
    def forward(self, x):
        Kernel = 4 * self.Kernel.view(1, 1, self.Kernel.shape[0], self.Kernel.shape[1]).to(x.dtype)
        y = nn.functional.conv_transpose2d(x.view(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3]), Kernel, stride=2, padding=self.FilterRadius)
        
        return y.view(x.shape[0], x.shape[1], y.shape[2], y.shape[3])
    
class InterpolativeDownsamplerReference(nn.Module):
    def __init__(self, Filter):
        super(InterpolativeDownsamplerReference, self).__init__()
        
        self.register_buffer('Kernel', CreateLowpassKernel(Filter, Inplace=False))
        self.FilterRadius = len(Filter) // 2
        
    def forward(self, x):
        Kernel = self.Kernel.view(1, 1, self.Kernel.shape[0], self.Kernel.shape[1]).to(x.dtype)
        y = nn.functional.conv2d(x.view(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3]), Kernel, stride=2, padding=self.FilterRadius)
        
        return y.view(x.shape[0], x.shape[1], y.shape[2], y.shape[3])
    
class InplaceUpsamplerReference(nn.Module):
    def __init__(self, Filter):
        super(InplaceUpsamplerReference, self).__init__()
        
        self.register_buffer('Kernel', CreateLowpassKernel(Filter, Inplace=True))
        self.FilterRadius = len(Filter) // 2
        
    def forward(self, x):
        Kernel = self.Kernel.view(1, 1, self.Kernel.shape[0], self.Kernel.shape[1]).to(x.dtype)
        x = nn.functional.pixel_shuffle(x, 2)
        
        return nn.functional.conv2d(x.view(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3]), Kernel, stride=1, padding=self.FilterRadius).view(*x.shape)
    
class InplaceDownsamplerReference(nn.Module):
    def __init__(self, Filter):
        super(InplaceDownsamplerReference, self).__init__()
        
        self.register_buffer('Kernel', CreateLowpassKernel(Filter, Inplace=True))
        self.FilterRadius = len(Filter) // 2
        
    def forward(self, x):
        Kernel = self.Kernel.view(1, 1, self.Kernel.shape[0], self.Kernel.shape[1]).to(x.dtype)
        y = nn.functional.conv2d(x.view(x.shape[0] * x.shape[1], 1, x.shape[2], x.shape[3]), Kernel, stride=1, padding=self.FilterRadius).view(*x.shape)
        
        return nn.functional.pixel_unshuffle(y, 2)
    
class InterpolativeUpsamplerCUDA(nn.Module):
    def __init__(self, Filter):
        super(InterpolativeUpsamplerCUDA, self).__init__()
        
        self.register_buffer('Kernel', CreateLowpassKernel(Filter, Inplace=False))
        
    def forward(self, x):
        return upfirdn2d.upsample2d(x, self.Kernel)

class InterpolativeDownsamplerCUDA(nn.Module):
    def __init__(self, Filter):
        super(InterpolativeDownsamplerCUDA, self).__init__()
        
        self.register_buffer('Kernel', CreateLowpassKernel(Filter, Inplace=False))
        
    def forward(self, x):
        return upfirdn2d.downsample2d(x, self.Kernel)

class InplaceUpsamplerCUDA(nn.Module):
    def __init__(self, Filter):
        super(InplaceUpsamplerCUDA, self).__init__()
        
        self.register_buffer('Kernel', CreateLowpassKernel(Filter, Inplace=True))
        self.FilterRadius = len(Filter) // 2
        
    def forward(self, x):
        return upfirdn2d.upfirdn2d(nn.functional.pixel_shuffle(x, 2), self.Kernel, padding=self.FilterRadius)

class InplaceDownsamplerCUDA(nn.Module):
    def __init__(self, Filter):
        super(InplaceDownsamplerCUDA, self).__init__()
        
        self.register_buffer('Kernel', CreateLowpassKernel(Filter, Inplace=True))
        self.FilterRadius = len(Filter) // 2
        
    def forward(self, x):
        return nn.functional.pixel_unshuffle(upfirdn2d.upfirdn2d(x, self.Kernel, padding=self.FilterRadius), 2)

InterpolativeUpsampler = InterpolativeUpsamplerCUDA
InterpolativeDownsampler = InterpolativeDownsamplerCUDA
InplaceUpsampler = InplaceUpsamplerCUDA
InplaceDownsampler = InplaceDownsamplerCUDA