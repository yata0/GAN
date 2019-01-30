import torch
import torch.nn.functional as F
import torch.nn as nn

def conv(in_channels, out_channels, kernel_size, stride=2, padding=1,batch_norm=True):

    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                    kernel_size=kernel_size, stride=stride,padding=padding, bias=False)
    layers.append(conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

def deconv(in_channels, out_channels, kernel_size, stride=2,padding=1,batch_norm=True):
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,padding, bias=False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)



class ResidualBlock(nn.Module):

    def __init__(self, conv_dim):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv(conv_dim, conv_dim,kernel_size=3,stride=1)
        self.conv2 = conv(conv_dim, conv_dim, kernel_size=3, stride=1)
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += x
        return x

class Discriminator(nn.Module):
    def __init__(self, conv_dim=64):
        super(Discriminator, self).__init__()
        self.conv1 = conv(3,conv_dim,4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 = conv(conv_dim*2,conv_dim*4,4)
        self.conv4 = conv(conv_dim*4,conv_dim*8,4)
        self.conv5 = conv(conv_dim*8,1,4,1,batch_norm=False)
    def forward(self,x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.conv5(out)
        return out

class CycleGenerator(nn.Module):
    def __init__(self, conv_dim=64,n_res_blocks=6):
        super(CycleGenerator,self).__init__()
        self.encoder_conv1 = conv(3,conv_dim,4)
        self.encoder_conv2 = conv(conv_dim, conv_dim*2,4)
        self.encoder_conv3 = conv(conv_dim*2,conv_dim*4,4)
        residual_layers = []
        for _ in range(n_res_blocks):
            residual_layers.append(ResidualBlock(conv_dim*4))
        self.residual_layers = nn.Sequential(*residual_layers)

        self.decoder_conv1 = deconv(conv_dim*4, conv_dim*2, 4)
        self.decoder_conv2 = deconv(conv_dim*2, conv_dim, 4)
        self.decoder_conv3 = deconv(conv_dim, 3, 4, batch_norm=False)
    
    def forward(self, x):
        out = F.relu(self.encoder_conv1(x))
        out = F.relu(self.encoder_conv2(out))
        out = F.relu(self.encoder_conv3(out))
        out = self.residual_layers(out)
        out = F.relu(self.decoder_conv1(out))
        out = F.relu(self.decoder_conv2(out))
        out = F.tanh(self.decoder_conv3(out))
        return out

def create_model(g_conv_dim=64,d_conv_dim=64,n_res_blocks=6):
    G_XtoY = CycleGenerator(g_conv_dim, n_res_blocks)
    G_YtoX = CycleGenerator(g_conv_dim, n_res_blocks)
    D_X = Discriminator(d_conv_dim)
    D_Y = Discriminator(d_conv_dim)

    if torch.cuda.is_available():
        # device = torch.device("cuda")
        G_XtoY = G_XtoY.cuda()
        G_YtoX = G_YtoX.cuda()
        D_X = D_X.cuda()
        D_Y = D_Y.cuda()
        print("Models move to GPU")
    else:
        print("Only CPU avaliable")
    return G_XtoY,G_YtoX,D_X,D_Y

def print_structure():
    G_XtoY,G_YtoX,D_X,D_Y = create_model()
    test = torch.zeros([1,3,128,128])
    if torch.cuda.is_available():
        test = test.cuda()
    print("input shape:")
    print(test.shape)
    print("\t\t\tGenerator X to Y\t\t\t")
    print("--"*30)

    print(G_XtoY)
    print("output")
    print(G_XtoY.forward(test).shape)
    print()
    print("\t\t\tGenerator Y to X\t\t\t")
    print("--"*30)
    print()
    print(G_YtoX)
    print("output")
    print(G_YtoX.forward(test).shape)
    print("\t\t\tDiscriminator X\t\t\t")
    print("--"*30)
    print(D_X)
    print("output")
    print(D_X.forward(test).shape)
    print("\t\t\tDiscriminator Y\t\t\t")
    print("--"*30)
    print(D_Y)
    print("output")
    print(D_Y.forward(test).shape)
    
    

if __name__ == "__main__":
    print_structure()

