import math
import torch
import torch.nn as nn
import math
# 初始化权重
def MSRInitializer(Layer, ActivationGain=1):
    FanIn = Layer.weight.data.size(1) * Layer.weight.data[0][0].numel()
    Layer.weight.data.normal_(0,  ActivationGain / math.sqrt(FanIn))

    if Layer.bias is not None:
        Layer.bias.data.zero_()
    
    return Layer


# 卷积层
class Convolution(nn.Module):
    def __init__(self, InputChannels, OutputChannels, KernelSize, Groups=1, ActivationGain=1):
        super(Convolution, self).__init__()
        self.Layer = MSRInitializer(nn.Conv1d(InputChannels, OutputChannels, kernel_size=KernelSize, stride=1, padding=(KernelSize - 1) // 2, groups=Groups, bias=False), ActivationGain=ActivationGain)
        
    def forward(self, x):
        return nn.functional.conv1d(x, self.Layer.weight.to(x.dtype), padding=self.Layer.padding, groups=self.Layer.groups)

# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, InputChannels, Cardinality, ExpansionFactor, KernelSize, VarianceScalingParameter):
        super(ResidualBlock, self).__init__()
        NumberOfLinearLayers = 3
        ExpandedChannels = InputChannels * ExpansionFactor
        ActivationGain = 1 * VarianceScalingParameter ** (-1 / (2 * NumberOfLinearLayers - 2))  # 假设 BiasedActivation.Gain 为 1

        self.LinearLayer1 = Convolution(InputChannels, ExpandedChannels, KernelSize=1, ActivationGain=ActivationGain)
        self.LinearLayer2 = Convolution(ExpandedChannels, ExpandedChannels, KernelSize=KernelSize, Groups=Cardinality, ActivationGain=ActivationGain)
        self.LinearLayer3 = Convolution(ExpandedChannels, InputChannels, KernelSize=1, ActivationGain=0)
        
        self.NonLinearity1 = nn.LeakyReLU(0.2)
        self.NonLinearity2 = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        y = self.LinearLayer1(x)
        y = self.LinearLayer2(self.NonLinearity1(y))
        y = self.LinearLayer3(self.NonLinearity2(y))
        
        return x + y

# 上采样层
class UpsampleLayer(nn.Module):
    def __init__(self, InputChannels, OutputChannels):
        super(UpsampleLayer, self).__init__()
        if InputChannels != OutputChannels:
            self.LinearLayer = Convolution(InputChannels, OutputChannels, KernelSize=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        
    def forward(self, x):
        x = self.LinearLayer(x) if hasattr(self, 'LinearLayer') else x
        x = self.upsample(x)
        return x

# 下采样层
class DownsampleLayer(nn.Module):
    def __init__(self, InputChannels, OutputChannels):
        super(DownsampleLayer, self).__init__()
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)
        if InputChannels != OutputChannels:
            self.LinearLayer = Convolution(InputChannels, OutputChannels, KernelSize=1)
        
    def forward(self, x):
        x = self.downsample(x)
        x = self.LinearLayer(x) if hasattr(self, 'LinearLayer') else x
        return x

# 生成器基础层
class GenerativeBasis(nn.Module):
    def __init__(self, InputDimension, OutputChannels, seq_len):
        super(GenerativeBasis, self).__init__()
        self.Basis = nn.Parameter(torch.empty(OutputChannels, seq_len).normal_(0, 1))
        self.LinearLayer = MSRInitializer(nn.Linear(InputDimension, OutputChannels, bias=False))
        
    def forward(self, x):
        return self.Basis.view(1, -1, self.Basis.shape[1]) * self.LinearLayer(x).view(x.shape[0], -1, 1)

# 判别器基础层
class DiscriminativeBasis(nn.Module):
    def __init__(self, InputChannels, OutputDimension, seq_len):
        super(DiscriminativeBasis, self).__init__()
        self.Basis = MSRInitializer(nn.Conv1d(InputChannels, InputChannels, kernel_size=seq_len, stride=1, padding=0, groups=InputChannels, bias=False))
        self.LinearLayer = MSRInitializer(nn.Linear(InputChannels, OutputDimension, bias=False))
        
    def forward(self, x):
        return self.LinearLayer(self.Basis(x).view(x.shape[0], -1))

# 生成器阶段
class GeneratorStage(nn.Module):
    def __init__(self, InputChannels, OutputChannels, Cardinality, NumberOfBlocks, ExpansionFactor, KernelSize, VarianceScalingParameter, seq_len, is_first_stage=False):
        super(GeneratorStage, self).__init__()
        TransitionLayer = GenerativeBasis(InputChannels, OutputChannels, seq_len) if is_first_stage else UpsampleLayer(InputChannels, OutputChannels)
        self.Layers = nn.ModuleList([TransitionLayer] + [ResidualBlock(OutputChannels, Cardinality, ExpansionFactor, KernelSize, VarianceScalingParameter) for _ in range(NumberOfBlocks)])
        
    def forward(self, x):
        for Layer in self.Layers:
            x = Layer(x)
        return x

# 判别器阶段
class DiscriminatorStage(nn.Module):
    def __init__(self, InputChannels, OutputChannels, Cardinality, NumberOfBlocks, ExpansionFactor, KernelSize, VarianceScalingParameter, seq_len, is_last_stage=False):
        super(DiscriminatorStage, self).__init__()
        TransitionLayer = DiscriminativeBasis(InputChannels, OutputChannels, seq_len) if is_last_stage else DownsampleLayer(InputChannels, OutputChannels)
        self.Layers = nn.ModuleList([ResidualBlock(InputChannels, Cardinality, ExpansionFactor, KernelSize, VarianceScalingParameter) for _ in range(NumberOfBlocks)] + [TransitionLayer])
        
    def forward(self, x):
        for Layer in self.Layers:
            x = Layer(x)
        return x

# 生成器
class Generator(nn.Module):
    def __init__(self, configs):
        super(Generator, self).__init__()
        NoiseDimension = 64
        WidthPerStage = [4 * x //3 for x in [256, 256, 256, 256]]
        BlocksPerStage = [2 * x for x in [1, 1, 1, 1]]
        CardinalityPerStage = [4 * x for x in [24, 24, 24, 24]]
        ExpansionFactor = 4
        KernelSize = 3
        seq_len = configs.seq_len

        VarianceScalingParameter = sum(BlocksPerStage)
        MainLayers = [GeneratorStage(NoiseDimension, WidthPerStage[0], CardinalityPerStage[0], BlocksPerStage[0], ExpansionFactor, KernelSize, VarianceScalingParameter, seq_len, is_first_stage=True)]
        MainLayers += [GeneratorStage(WidthPerStage[x], WidthPerStage[x + 1], CardinalityPerStage[x + 1], BlocksPerStage[x + 1], ExpansionFactor, KernelSize, VarianceScalingParameter, seq_len) for x in range(len(WidthPerStage) - 1)]
        
        self.MainLayers = nn.ModuleList(MainLayers)
        self.AggregationLayer = Convolution(WidthPerStage[-1], configs.enc_in, KernelSize=1)
        
    def forward(self, x):
        for Layer in self.MainLayers:
            x = Layer(x)
        return self.AggregationLayer(x)

# 判别器
class Discriminator(nn.Module):
    def __init__(self, configs):
        super(Discriminator, self).__init__()
        WidthPerStage = [4 * x //3 for x in [256, 256, 256, 256]]
        BlocksPerStage = [2 * x for x in [1, 1, 1, 1]]
        CardinalityPerStage = [4 * x for x in [24, 24, 24, 24]]
        ExpansionFactor = 4
        KernelSize = 3
        seq_len = configs.seq_len

        VarianceScalingParameter = sum(BlocksPerStage)
        MainLayers = [DiscriminatorStage(WidthPerStage[x], WidthPerStage[x + 1], CardinalityPerStage[x], BlocksPerStage[x], ExpansionFactor, KernelSize, VarianceScalingParameter, seq_len) for x in range(len(WidthPerStage) - 1)]
        MainLayers += [DiscriminatorStage(WidthPerStage[-1], 1, CardinalityPerStage[-1], BlocksPerStage[-1], ExpansionFactor, KernelSize, VarianceScalingParameter, seq_len, is_last_stage=True)]
        
        self.ExtractionLayer = Convolution(configs.enc_in, WidthPerStage[0], KernelSize=1)
        self.MainLayers = nn.ModuleList(MainLayers)
        
    def forward(self, x):
        x = self.ExtractionLayer(x)
        for Layer in self.MainLayers:
            x = Layer(x)
        return x.view(x.shape[0])

# 完整的GAN模型
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.generator = Generator(configs)
        self.discriminator = Discriminator(configs)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 生成随机噪声
        z = torch.randn(x_enc.size(0), self.configs.noise_dim).to(x_enc.device)
        # 生成数据
        generated_data = self.generator(z)
        return generated_data