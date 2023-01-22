import torch.nn as nn
import torch.nn.functional as F
#import torch
#import torch.optim as optim
#import pytorch_lightning as pl
from .dss import DSS

from .pos_encoder import PositionalEncoder
import hydra

class L1Block(nn.Module):

    def __init__(self,channels=256,kernel_size=7):
        super(L1Block, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=channels,out_channels=channels,padding=int((kernel_size-1)/2),kernel_size=kernel_size,bias=True)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(in_channels=channels,out_channels=channels,padding=int((kernel_size-1)/2),kernel_size=kernel_size,bias=True)
        self.bn2 = nn.BatchNorm1d(channels)
        self.layer = nn.Sequential(self.conv1, self.bn1, nn.ReLU(inplace=True), self.conv2, self.bn2)

    def forward(self, x):
        out = self.layer(x)
        out += x
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(
        self, 
        d_input=5, 
        d_output=5, 
        d_model=512, 
        n_layers = 5,
        pe_when=None,
        pe=None,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        self.encoder = nn.Conv1d(in_channels=d_input,out_channels=d_model,padding=int((31-1)/2),kernel_size=31,bias=True)

        # positional encoding,
        if pe is not None:
            self.pe = hydra.utils.instantiate(pe)
            assert pe_when in ["first", "after_conv"], "Unknown configuration for pe_when"
            self.pe_when = pe_when
        else:
            self.pe = None
            self.pe_when = None

        self.decoder = nn.Conv1d(in_channels=d_model,out_channels=d_output,padding=int((31-1)/2),kernel_size=31,bias=True)

        self.resnet_layer = nn.Sequential(*[L1Block(channels=d_model) for x in range(n_layers)])

    def forward(self, x):
        """
        Input x is shape (B, d_input, L)
        """

        #[1024, 5, 604])

        #import pdb;pdb.set_trace()
        #x = x.squeeze()

        # positional encoding
        if self.pe_when is not None and self.pe_when=="first":
            # expects (L,B,dim)
            x = self.pe(x.permute((2,0,1))).permute(1,2,0) #.transpose(0,1)
        
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        # positional encoding after convolution
        if self.pe_when is not None and self.pe_when=="after_conv":
            # expects (L,B,dim)
            x = self.pe(x.permute((2,0,1))).permute(1,2,0) #.transpose(0,1)

        x = self.resnet_layer(x)
    
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)
        return x
