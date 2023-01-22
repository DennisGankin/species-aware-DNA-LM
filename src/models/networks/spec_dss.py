
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torch.nn.functional as F
import numpy as np

from .dss import DSS

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

class SpecAdd(nn.Module):
    def __init__(self, d_model, embed, encoder="label", k=5) -> None:
        super().__init__()

        #if encoder=="label":
        self.species_embedder = nn.Embedding(806,d_model)
        self.embed = embed

    def forward(self, x, spec_labels):

        if self.embed:
            # adding to each token

            spec_labels = self.species_embedder(spec_labels)

            x = (x.transpose(0,-1) + spec_labels.transpose(0,1)).transpose(0,-1)

            return x
        else:
            # concatenating
            spec_labels = self.species_embedder(spec_labels)[:,:, None] # ()
            torch.cat((x,spec_labels), dim = -1)

            return x

class DSSResNet(nn.Module):
    def __init__(
        self, 
        d_input=5, 
        d_output=5, 
        d_model=128, 
        n_layers=4, 
        dropout=0.2,
        prenorm=False,
        species_encoder = None,
        embed_before = False,
        #embed='label'
    ):
        super().__init__()

        self.prenorm = prenorm

        self.encoder = nn.Conv1d(in_channels=d_input,out_channels=d_model,padding=int((15-1)/2),kernel_size=15,bias=True)

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

        self.decoder = nn.Conv1d(in_channels=d_model,out_channels=d_output,padding=int((15-1)/2),kernel_size=15,bias=True)

        self.resnet_layer = nn.Sequential(*[L1Block(channels=d_model) for x in range(3)])

        self.species_encoder = species_encoder
        self.embed_before = embed_before

    def forward(self, x, xs):
        """
        Input x is shape (B, d_input, L)
        """

        x = self.encoder(x)  # (B, d_input, L) -> (B, d_model, L)

        if self.embed_before:
            x = self.species_encoder(x,xs)

        #print("shape after encoder", x.size())
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

        if not self.embed_before:
            x = self.species_encoder(x,xs)

        x = self.decoder(x)  # (B, d_model, L) -> (B, d_output, L)

        return x


class DSSResNetEmb(nn.Module):
    def __init__(
        self, 
        d_input=5, 
        d_output=5, 
        d_model=128, 
        n_layers=4, 
        dropout=0.2,
        prenorm=False,
        species_encoder = None,
        embed_before = False,
        #embed='label'
    ):
        super().__init__()

        self.prenorm = prenorm
        self.encoder = nn.Conv1d(in_channels=d_input,out_channels=d_model,padding=int((15-1)/2),kernel_size=15,bias=True)

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

        self.decoder = nn.Conv1d(in_channels=d_model,out_channels=d_output,padding=int((15-1)/2),kernel_size=15,bias=True)

        self.resnet_layer = nn.Sequential(*[L1Block(channels=d_model) for x in range(3)])

        self.species_encoder = species_encoder
        self.embed_before = embed_before

    def forward(self, x, xs):
        """
        Input x is shape (B, d_input, L)
        """

        x = self.encoder(x)  # (B, d_input, L) -> (B, d_model, L)

        if self.embed_before:
            x = self.species_encoder(x,xs)

        #print("shape after encoder", x.size())
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

        
        if not self.embed_before:
            x = self.species_encoder(x,xs)

        seq_embedding = x

        x = self.decoder(x)  # (B, d_model, L) -> (B, d_output, L)

        embeddings = {}
        embeddings["seq_embedding"] = seq_embedding

        return x, embeddings





class DSSResNet2(nn.Module):
    def __init__(
        self, 
        d_input=5, 
        d_output=5, 
        d_model=128, 
        n_layers=4, 
        dropout=0.2,
        kernel_size=15,
        resnet_kernel_size=21,
        n_resnet_layers=3,
        prenorm=False,
        species_encoder = None,
        embed_before = False,
        #embed='label'
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        #self.encoder = nn.Linear(d_input, d_model) 
        self.encoder = nn.Conv1d(in_channels=d_input,out_channels=d_model,padding=int((kernel_size-1)/2),kernel_size=kernel_size,bias=True)

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


        self.decoder = nn.Conv1d(in_channels=d_model,out_channels=d_output,padding=int((kernel_size-1)/2),kernel_size=kernel_size,bias=True)

        #self.resnet_layer = nn.Sequential(*[L1Block(channels=d_model) for x in range(3)])
        self.resnet_layer = nn.Sequential(*[L1Block(channels=d_model, kernel_size=resnet_kernel_size+i*10) for i in range(n_resnet_layers)])
        self.resnet_layer2 = nn.Sequential(*[L1Block(channels=d_model, kernel_size=(n_resnet_layers-1)*10+resnet_kernel_size-i*10) for i in range(n_resnet_layers)])



        self.species_encoder = species_encoder
        self.embed_before = embed_before

    def forward(self, x, xs):
        """
        Input x is shape (B, d_input, L)
        """

        x = self.encoder(x)  # (B, d_input, L) -> (B, d_model, L)

        if self.embed_before:
            x = self.species_encoder(x,xs)

        #print("shape after encoder", x.size())
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

        if not self.embed_before:
            x = self.species_encoder(x,xs)

        x = self.resnet_layer2(x)

        x = self.decoder(x)  # (B, d_model, L) -> (B, d_output, L)

        return x
