import torch
import torch.nn as nn
import torch.nn.functional as F

# python 3 confusing imports :(
from unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, shrink):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64 // shrink)
        self.down1 = down(64 // shrink, 128 // shrink)
        self.down2 = down(128 // shrink, 256 // shrink)
        self.down3 = down(256 // shrink, 512 // shrink)
        self.down4 = down(512 // shrink, 512 // shrink)
        self.up1 = up(1024 // shrink, 256 // shrink)
        self.up2 = up(512 // shrink, 128 // shrink)
        self.up3 = up(256 // shrink, 64 // shrink)
        self.up4 = up(128 // shrink, 64 // shrink)

    def forward(self, x):
        x1 = self.inc(x)
        #print("x1 shape",x1.shape)
        x2 = self.down1(x1)
        #print("x2 shape",x2.shape)
        x3 = self.down2(x2)
        #print("x3 shape",x3.shape)

        x4 = self.down3(x3)
        x5 = self.down4(x4)
#         out1 = self.up1(x5, x4)
#         out2 = self.up2(out1, x3)
#         out3 = self.up3(out2, x2)
        return x5
class Feedforward(nn.Module):
    def __init__(self,input_size, hidden_size=4096,final=2048):
        super(Feedforward, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, final)  

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x        