import math
import torch
import torch.nn as nn

from pytorch_lightning import LightningModule

"""
Positional encoding from https://github.com/pytorch/examples/blob/main/word_language_model/model.py
"""
class PositionalEncodingold(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.0, max_len=2000):
        super(PositionalEncodingold, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, step=2).float() * (-math.log(10000.0) / d_model))
        print(div_term.shape)
        print(position.shape)
        tmp = position * div_term
        pe[:, 0::2] = torch.sin(tmp)
        print(pe[:, 1::2].shape)
        print(tmp.shape)
        pe[:, 1::2] = torch.cos(tmp)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# fix from https://discuss.pytorch.org/t/transformer-example-position-encoding-function-works-only-for-even-d-model/100986
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model%2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:,0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def pe_absolute(max_len):
    """Creates absolute positional embedding

    Args:
        max_len (int): max sequence length
        d_model (int): model dimension, not needed

    Returns:
        torch.tensor: positional embedding tensor
    """

    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

    # L, 1, 1
    return position.unsqueeze(0).transpose(0, 1)

def pe_relative(max_len):
    """Creates relative positional embedding

    Args:
        max_len (int): max sequence length
        d_model (int): model dimension, not needed

    Returns:
        torch.tensor: positional embedding tensor
    """

    return pe_absolute(max_len)/max_len

def pe_sinusoidal(max_len, d_model):
    """Creates sinusoidal positional embedding
    from Positional encoding from https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    and  fix from https://discuss.pytorch.org/t/transformer-example-position-encoding-function-works-only-for-even-d-model/100986

    Args:
        max_len (int): max sequence length
        d_model (int): model dimension

    Returns:
        torch.tensor: positional embedding tensor
    """

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    # make sure it works with odd and even lengths
    if d_model%2 != 0:
        pe[:, 1::2] = torch.cos(position * div_term)[:,0:-1]
    else:
        pe[:, 1::2] = torch.cos(position * div_term)
    # L, 1, dim
    pe = pe.unsqueeze(0).transpose(0, 1)

    return pe

class PositionalEncoder(LightningModule):
    """Abstract class"""
    
    def __init__(self, dropout=0.1, how="add"):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # create PE based on method

        # save as non-trainable params 
        #self.register_buffer('pe', pe)
        self.register_buffer('p',torch.zeros(5))
        self.how = how

    def forward(self,x):
        # either append or add
        if self.how=="add":
            x = x + self.pe[:x.size(0), :] 
        elif self.how=="stack":
            p = torch.zeros(x.size(0),x.size(1),1)
            #print(p)
            #print(x)
            p = self.pe[:x.size(0)]
            x = torch.cat((x.transpose(0,-1),p.transpose(0,-1))).transpose(0,-1)
        elif self.how == "concat":
            p = torch.zeros(x.size()) + self.pe[:x.size(0), :]
            x = torch.cat((x,p))
        else:
            raise NotImplementedError("The operator is unknown:" + self.how)
        return self.dropout(x)


class SinusoidalPE(PositionalEncoder):

    def __init__(self, d_model=5, how="add", dropout=0.1, max_len=5000):
        super().__init__(dropout,how)

        pe = pe_sinusoidal(max_len, d_model)
        self.register_buffer('pe', pe)
        
class AbsolutePE(PositionalEncoder):
    def __init__(self, d_model=5, how="add", dropout=0.1, max_len=5000):
        super().__init__(dropout,how)

        pe = pe_absolute(max_len)
        self.register_buffer('pe', pe)

class RelativePE(PositionalEncoder):

    def __init__(self, d_model=5, how="add", dropout=0.1, max_len=5000):
        super().__init__(dropout,how)

        pe = pe_relative(max_len)
        self.register_buffer('pe', pe)

