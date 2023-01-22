import torch
from torch.nn.functional import one_hot
import numpy as np

class OneHot(torch.nn.Module):
    def __init__(self, num_classes = 805) -> None:
        super().__init__()

        self.num_classes = num_classes

    def forward(self, x):

        return one_hot(x, self.num_classes)


