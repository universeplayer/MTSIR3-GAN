import torch
import torch.nn as nn
import math
from torch_utils.ops import bias_act

class BiasedActivationReference(nn.Module):
    Gain = math.sqrt(2 / (1 + 0.2 ** 2))
    Function = nn.LeakyReLU(0.2)
    
    def __init__(self, InputUnits):
        super(BiasedActivationReference, self).__init__()
        
        self.Bias = nn.Parameter(torch.empty(InputUnits))
        self.Bias.data.zero_()
        
    def forward(self, x):
        y = x + self.Bias.to(x.dtype).view(1, -1, 1, 1) if len(x.shape) > 2 else x + self.Bias.to(x.dtype).view(1, -1)
        return BiasedActivationReference.Function(y)

class BiasedActivationCUDA(nn.Module):
    Gain = math.sqrt(2 / (1 + 0.2 ** 2))
    Function = 'lrelu'
    
    def __init__(self, InputUnits):
        super(BiasedActivationCUDA, self).__init__()
        
        self.Bias = nn.Parameter(torch.empty(InputUnits))
        self.Bias.data.zero_()
        
    def forward(self, x):
        return bias_act.bias_act(x, self.Bias.to(x.dtype), act=BiasedActivationCUDA.Function, gain=1)

BiasedActivation = BiasedActivationCUDA