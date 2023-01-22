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

class DSSResNet(nn.Module):
    def __init__(
        self, 
        d_input=5, 
        d_output=5, 
        d_model=128, 
        n_layers=4, 
        dropout=0.2,
        kernel_size=15,
        resnet_kernel_size=41,
        n_resnet_layers=3,
        pe_when=None,
        pe=None,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        #self.encoder = nn.Linear(d_input, d_model) 
        self.encoder = nn.Conv1d(in_channels=d_input,out_channels=d_model,padding=int((kernel_size-1)/2),kernel_size=kernel_size,bias=True)

        # positional encoding
        if pe is not None:
            self.pe = hydra.utils.instantiate(pe)
            assert pe_when in ["first", "after_conv"], "Unknown configuration for pe_when"
            self.pe_when = pe_when
        else:
            self.pe = None
            self.pe_when = None

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                DSS(
                    d_model=d_model, 
                    l_max=1, 
                    bidirectional=True,
                    postact='glu',
                    dropout=dropout, 
                    transposed=True,
                )
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout2d(dropout))

        # Linear decoder
        # self.decoder = nn.Linear(d_model, d_output)
        self.decoder = nn.Conv1d(in_channels=d_model,out_channels=d_output,padding=int((kernel_size-1)/2),kernel_size=kernel_size,bias=True)

        self.resnet_layer = nn.Sequential(*[L1Block(channels=d_model) for i in range(n_resnet_layers)]) #, kernel_size=resnet_kernel_size+i*10) for i in range(n_resnet_layers)])
        #self.resnet_layer2 = nn.Sequential(*[L1Block(channels=d_model, kernel_size=(n_resnet_layers-1)*10+resnet_kernel_size-i*10) for i in range(n_resnet_layers)])
        #+ (n_resnet_layers-1)*10+

    def forward(self, x):
        """
        Input x is shape (B, d_input, L)
        """

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
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            
            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Decode the outputs
        x = x.transpose(-1, -2)

        x = self.decoder(x)  # (B, d_model) -> (B, d_output)
        return x
