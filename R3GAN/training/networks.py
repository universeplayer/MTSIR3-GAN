import torch
import torch.nn as nn
import copy
import R3GAN.Networks

class Generator(nn.Module):
    def __init__(self, *args, **kw):
        super(Generator, self).__init__()
        
        config = copy.deepcopy(kw)
        del config['FP16Stages']
        del config['c_dim']
        del config['img_resolution']
        
        if kw['c_dim'] != 0:
            config['ConditionDimension'] = kw['c_dim']
        
        self.Model = R3GAN.Networks.Generator(*args, **config)
        self.z_dim = kw['NoiseDimension']
        self.c_dim = kw['c_dim']
        self.img_resolution = kw['img_resolution']
        
        for x in kw['FP16Stages']:
            self.Model.MainLayers[x].DataType = torch.bfloat16
        
    def forward(self, x, c):
        return self.Model(x, c)
    
class Discriminator(nn.Module):
    def __init__(self, *args, **kw):
        super(Discriminator, self).__init__()
        
        config = copy.deepcopy(kw)
        del config['FP16Stages']
        del config['c_dim']
        del config['img_resolution']
        
        if kw['c_dim'] != 0:
            config['ConditionDimension'] = kw['c_dim']
        
        self.Model = R3GAN.Networks.Discriminator(*args, **config)
        
        for x in kw['FP16Stages']:
            self.Model.MainLayers[x].DataType = torch.bfloat16
        
    def forward(self, x, c):
        return self.Model(x, c)