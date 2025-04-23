import math
import torch
import torch.nn as nn
from .Resamplers import InterpolativeUpsampler, InterpolativeDownsampler
from .FusedOperators import BiasedActivation

def MSRInitializer(Layer, ActivationGain=1):
    FanIn = Layer.weight.data.size(1) * Layer.weight.data[0][0].numel()
    Layer.weight.data.normal_(0,  ActivationGain / math.sqrt(FanIn))

    if Layer.bias is not None:
        Layer.bias.data.zero_()
    
    return Layer

class Convolution(nn.Module):
    def __init__(self, InputChannels, OutputChannels, KernelSize, Groups=1, ActivationGain=1):
        super(Convolution, self).__init__()
        
        self.Layer = MSRInitializer(nn.Conv2d(InputChannels, OutputChannels, kernel_size=KernelSize, stride=1, padding=(KernelSize - 1) // 2, groups=Groups, bias=False), ActivationGain=ActivationGain)
        
    def forward(self, x):
        return nn.functional.conv2d(x, self.Layer.weight.to(x.dtype), padding=self.Layer.padding, groups=self.Layer.groups)

class ResidualBlock(nn.Module):
    def __init__(self, InputChannels, Cardinality, ExpansionFactor, KernelSize, VarianceScalingParameter):
        super(ResidualBlock, self).__init__()
        
        NumberOfLinearLayers = 3
        ExpandedChannels = InputChannels * ExpansionFactor
        ActivationGain = BiasedActivation.Gain * VarianceScalingParameter ** (-1 / (2 * NumberOfLinearLayers - 2))
        
        self.LinearLayer1 = Convolution(InputChannels, ExpandedChannels, KernelSize=1, ActivationGain=ActivationGain)
        self.LinearLayer2 = Convolution(ExpandedChannels, ExpandedChannels, KernelSize=KernelSize, Groups=Cardinality, ActivationGain=ActivationGain)
        self.LinearLayer3 = Convolution(ExpandedChannels, InputChannels, KernelSize=1, ActivationGain=0)
        
        self.NonLinearity1 = BiasedActivation(ExpandedChannels)
        self.NonLinearity2 = BiasedActivation(ExpandedChannels)
        
    def forward(self, x):
        y = self.LinearLayer1(x)
        y = self.LinearLayer2(self.NonLinearity1(y))
        y = self.LinearLayer3(self.NonLinearity2(y))
        
        return x + y
    
class UpsampleLayer(nn.Module):
    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        super(UpsampleLayer, self).__init__()
        
        self.Resampler = InterpolativeUpsampler(ResamplingFilter)
        
        if InputChannels != OutputChannels:
            self.LinearLayer = Convolution(InputChannels, OutputChannels, KernelSize=1)
        
    def forward(self, x):
        x = self.LinearLayer(x) if hasattr(self, 'LinearLayer') else x
        x = self.Resampler(x)
        
        return x
    
class DownsampleLayer(nn.Module):
    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        super(DownsampleLayer, self).__init__()
        
        self.Resampler = InterpolativeDownsampler(ResamplingFilter)
        
        if InputChannels != OutputChannels:
            self.LinearLayer = Convolution(InputChannels, OutputChannels, KernelSize=1)
        
    def forward(self, x):
        x = self.Resampler(x)
        x = self.LinearLayer(x) if hasattr(self, 'LinearLayer') else x
        
        return x
    
class GenerativeBasis(nn.Module):
    def __init__(self, InputDimension, OutputChannels):
        super(GenerativeBasis, self).__init__()
        
        self.Basis = nn.Parameter(torch.empty(OutputChannels, 4, 4).normal_(0, 1))
        self.LinearLayer = MSRInitializer(nn.Linear(InputDimension, OutputChannels, bias=False))
        
    def forward(self, x):
        return self.Basis.view(1, -1, 4, 4) * self.LinearLayer(x).view(x.shape[0], -1, 1, 1)
    
class DiscriminativeBasis(nn.Module):
    def __init__(self, InputChannels, OutputDimension):
        super(DiscriminativeBasis, self).__init__()
        
        self.Basis = MSRInitializer(nn.Conv2d(InputChannels, InputChannels, kernel_size=4, stride=1, padding=0, groups=InputChannels, bias=False))
        self.LinearLayer = MSRInitializer(nn.Linear(InputChannels, OutputDimension, bias=False))
        
    def forward(self, x):
        return self.LinearLayer(self.Basis(x).view(x.shape[0], -1))
    
class GeneratorStage(nn.Module):
    def __init__(self, InputChannels, OutputChannels, Cardinality, NumberOfBlocks, ExpansionFactor, KernelSize, VarianceScalingParameter, ResamplingFilter=None, DataType=torch.float32):
        super(GeneratorStage, self).__init__()
        
        TransitionLayer = GenerativeBasis(InputChannels, OutputChannels) if ResamplingFilter is None else UpsampleLayer(InputChannels, OutputChannels, ResamplingFilter)
        self.Layers = nn.ModuleList([TransitionLayer] + [ResidualBlock(OutputChannels, Cardinality, ExpansionFactor, KernelSize, VarianceScalingParameter) for _ in range(NumberOfBlocks)])
        self.DataType = DataType
        
    def forward(self, x):
        x = x.to(self.DataType)
        
        for Layer in self.Layers:
            x = Layer(x)
        
        return x
    
class DiscriminatorStage(nn.Module):
    def __init__(self, InputChannels, OutputChannels, Cardinality, NumberOfBlocks, ExpansionFactor, KernelSize, VarianceScalingParameter, ResamplingFilter=None, DataType=torch.float32):
        super(DiscriminatorStage, self).__init__()
        
        TransitionLayer = DiscriminativeBasis(InputChannels, OutputChannels) if ResamplingFilter is None else DownsampleLayer(InputChannels, OutputChannels, ResamplingFilter)
        self.Layers = nn.ModuleList([ResidualBlock(InputChannels, Cardinality, ExpansionFactor, KernelSize, VarianceScalingParameter) for _ in range(NumberOfBlocks)] + [TransitionLayer])
        self.DataType = DataType
        
    def forward(self, x):
        x = x.to(self.DataType)
        
        for Layer in self.Layers:
            x = Layer(x)
        
        return x
    
class Generator(nn.Module):
    def __init__(self, NoiseDimension, WidthPerStage, CardinalityPerStage, BlocksPerStage, ExpansionFactor, ConditionDimension=None, ConditionEmbeddingDimension=0, KernelSize=3, ResamplingFilter=[1, 2, 1]):
        super(Generator, self).__init__()
        
        VarianceScalingParameter = sum(BlocksPerStage)
        MainLayers = [GeneratorStage(NoiseDimension + ConditionEmbeddingDimension, WidthPerStage[0], CardinalityPerStage[0], BlocksPerStage[0], ExpansionFactor, KernelSize, VarianceScalingParameter)]
        MainLayers += [GeneratorStage(WidthPerStage[x], WidthPerStage[x + 1], CardinalityPerStage[x + 1], BlocksPerStage[x + 1], ExpansionFactor, KernelSize, VarianceScalingParameter, ResamplingFilter) for x in range(len(WidthPerStage) - 1)]
        
        self.MainLayers = nn.ModuleList(MainLayers)
        self.AggregationLayer = Convolution(WidthPerStage[-1], 3, KernelSize=1)
        
        if ConditionDimension is not None:
            self.EmbeddingLayer = MSRInitializer(nn.Linear(ConditionDimension, ConditionEmbeddingDimension, bias=False))
        
    def forward(self, x, y=None):
        x = torch.cat([x, self.EmbeddingLayer(y)], dim=1) if hasattr(self, 'EmbeddingLayer') else x
        
        for Layer in self.MainLayers:
            x = Layer(x)
        
        return self.AggregationLayer(x)
    
class Discriminator(nn.Module):
    def __init__(self, WidthPerStage, CardinalityPerStage, BlocksPerStage, ExpansionFactor, ConditionDimension=None, ConditionEmbeddingDimension=0, KernelSize=3, ResamplingFilter=[1, 2, 1]):
        super(Discriminator, self).__init__()
        
        VarianceScalingParameter = sum(BlocksPerStage)
        MainLayers = [DiscriminatorStage(WidthPerStage[x], WidthPerStage[x + 1], CardinalityPerStage[x], BlocksPerStage[x], ExpansionFactor, KernelSize, VarianceScalingParameter, ResamplingFilter) for x in range(len(WidthPerStage) - 1)]
        MainLayers += [DiscriminatorStage(WidthPerStage[-1], 1 if ConditionDimension is None else ConditionEmbeddingDimension, CardinalityPerStage[-1], BlocksPerStage[-1], ExpansionFactor, KernelSize, VarianceScalingParameter)]
        
        self.ExtractionLayer = Convolution(3, WidthPerStage[0], KernelSize=1)
        self.MainLayers = nn.ModuleList(MainLayers)
        
        if ConditionDimension is not None:
            self.EmbeddingLayer = MSRInitializer(nn.Linear(ConditionDimension, ConditionEmbeddingDimension, bias=False), ActivationGain=1 / math.sqrt(ConditionEmbeddingDimension))
        
    def forward(self, x, y=None):
        x = self.ExtractionLayer(x.to(self.MainLayers[0].DataType))
        
        for Layer in self.MainLayers:
            x = Layer(x)
        
        x = (x * self.EmbeddingLayer(y)).sum(dim=1, keepdim=True) if hasattr(self, 'EmbeddingLayer') else x
        
        return x.view(x.shape[0])
