import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
def conv(in_channels, out_channels, kernel_size, stride, padding,norm=True):
    layer = []
    conv_layer = nn.Conv2d(in_channels = in_channels, out_channels=out_channels,
                            kernel_size = kernel_size, stride=stride, padding=padding,bias=False)
    layer.append(conv_layer)
    if norm:
        layer.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layer)

def deconv(in_channels, out_channels, kernel_size, stride, padding, norm):
    layer = []
    deconv_layer = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size,stride=stride, padding=padding,bias=False)
    layer.append(deconv_layer)
    if norm:
        layer.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layer)

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(100,32*8*4*4)
        self.deconv1 = deconv(in_channels=32*8,out_channels=32*4,kernel_size=4,stride=2,padding=1,norm=True)
        self.deconv2 = deconv(in_channels=32*4,out_channels=32*2,kernel_size=4,stride=2,padding=1,norm=True)
        self.deconv3 = deconv(in_channels=32*2,out_channels=32,kernel_size=4,stride=2,padding=1,norm=True)
        self.deconv4 = deconv(in_channels=32,out_channels=3,kernel_size=4,stride=2,padding=1,norm=False)
    def forward(self, x):
        out = self.fc1(x)
        out = out.view(-1,32*8,4,4)
        out = F.relu(self.deconv1(out))
        out = F.relu(self.deconv2(out))
        out = F.relu(self.deconv3(out))
        out = F.tanh(self.deconv4(out))
        return out

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = conv(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1,norm=False)
        self.conv2 = conv(in_channels=32,out_channels=32*2,kernel_size=4, stride=2, padding=1,norm=True)
        
        self.conv3 = conv(in_channels=32*2, out_channels=32*4,kernel_size=4,stride=2, padding=1,norm=True)
        self.conv4 = conv(in_channels=32*4, out_channels=32*8,kernel_size=4,stride=2,padding=1,norm=True)
        self.output = nn.Linear(32*8*4*4,1)


    def forward(self, x):
        out = F.leaky_relu(self.conv1(x),negative_slope=0.2)
        out = F.leaky_relu(self.conv2(out), negative_slope=0.2)
        out = F.leaky_relu(self.conv3(out), negative_slope=0.2)
        out = F.leaky_relu(self.conv4(out), negative_slope=0.2)
        out = out.view(-1,32*8*4*4)
        out = self.output(out)
        return out
def init_weights(net, init_type="normal", gain=0.2):
    pass



        