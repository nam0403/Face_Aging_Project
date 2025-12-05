import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class Bottleneck(Module):
    def __init__(self, in_channel, depth, stride):
        super(Bottleneck, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), 
                BatchNorm2d(depth)
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth),
            SEModule(depth, 16)
        )
    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

class Backbone(Module):
    def __init__(self, num_layers=50, drop_ratio=0.4, mode='ir_se'):
        super(Backbone, self).__init__()
        assert num_layers in [50, 100, 152], "num_layers should be 50,100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = {50: [3, 4, 14, 3], 100: [3, 13, 30, 3], 152: [3, 8, 36, 3]}
        layers = blocks[num_layers] # [3, 4, 14, 3] for 50 layers

        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), (1, 1), 1, bias=False), 
                                      BatchNorm2d(64), PReLU(64))
        
        modules = []
        # Layer 1
        for i in range(layers[0]): modules.append(Bottleneck(64, 64, 2)) if i==0 else modules.append(Bottleneck(64, 64, 1))
        # Layer 2
        for i in range(layers[1]): modules.append(Bottleneck(64, 128, 2)) if i==0 else modules.append(Bottleneck(128, 128, 1))
        # Layer 3
        for i in range(layers[2]): modules.append(Bottleneck(128, 256, 2)) if i==0 else modules.append(Bottleneck(256, 256, 1))
        # Layer 4
        for i in range(layers[3]): modules.append(Bottleneck(256, 512, 2)) if i==0 else modules.append(Bottleneck(512, 512, 1))
        
        self.body = Sequential(*modules)
        
        self.output_layer = Sequential(BatchNorm2d(512), 
                                       Dropout(drop_ratio), 
                                       Flatten(), 
                                       Linear(512 * 7 * 7, 512), 
                                       BatchNorm1d(512))

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return x

def IR_SE50(embedding_size=512):
    """Hàm helper để gọi nhanh"""
    return Backbone(num_layers=50, mode='ir_se', drop_ratio=0.4)