"""Prediction baselines"""

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

# Make sure to use NLL loss for this
class ClassDist(LightningModule):
    def __init__(self, dist = [0.27161955, 0.21764797, 0.21911079, 0.29162169, 0.0000001], max_len=2000, max_batch=2000) -> None:
        super().__init__()

        # distribution (should contain probs for A, C , G , T , N)
        #dist = [0.0000001, 0.0000001, 0.0000001, 0.9999999, 0.0000001]

        # create tensor of max len
        self.prediction = torch.log(torch.tensor([[dist] * max_len]*max_batch).permute((1,2,0)))


    def forward(self, x):


        return self.prediction[:x.size(0),:,:x.size(2)].cuda()   #out.cuda()


class DiNucDist(LightningModule):
    def __init__(self, dinuc_for='train') -> None:
        super().__init__()

        # (should contain probs for A, C , G , T , N x A, C , G , T , N )
        # dinuc on train set
        train_dist = torch.log(torch.tensor([[0.28765218, 0.19895785, 0.21361906, 0.2997709 , 0.000001],
                                        [0.29905175, 0.2213963 , 0.19460597, 0.28494598, 0.000001],
                                        [0.29069304, 0.23104371, 0.22507592, 0.25318733, 0.000001],
                                        [0.22499235, 0.22467443, 0.24042836, 0.30990487, 0.000001],
                                        [0.000001, 0.000001, 0.000001, 0.000001, 0.000001]]).cuda())


        # dinuc overfit on Jun (all) SCer                                
        """
        scer_dist= torch.log(torch.tensor([[0.11650584, 0.05184092, 0.04646484, 0.12154135, 0.000001],
                                        [0.05492495, 0.0266681 , 0.02222659, 0.05321741, 0.000001],
                                        [0.04346159, 0.02781808, 0.02229628, 0.04956947, 0.000001],
                                        [0.11882639, 0.05400465, 0.05270894, 0.13792459, 0.000001],
                                        [0.000001, 0.000001, 0.000001, 0.000001, 0.000001]]).cuda())
        """
        # dinuc overfit on s.cer Jun
        scer_dist = torch.log(torch.tensor([[0.34761327, 0.15365545, 0.1374561 , 0.36127518, 0.000001],
                                        [0.35105206, 0.16913953, 0.13899755, 0.34081086, 0.000001],
                                        [0.30504691, 0.19255757, 0.15608539, 0.34631014, 0.000001],
                                        [0.32566986, 0.14829357, 0.1439468 , 0.38208977, 0.000001],
                                        [0.000001, 0.000001, 0.000001, 0.000001, 0.000001]]).cuda())

        # dinuc overfit on pombe Jun
        spombe_dist = torch.log(torch.tensor([[0.35827077, 0.14834849, 0.13990124, 0.3534795, 0.000001],
                                        [0.32813515, 0.16424929, 0.13672206, 0.37089349, 0.000001],
                                        [0.31019399, 0.18893581, 0.16364807, 0.33722212, 0.000001],
                                        [0.24572077, 0.15421273, 0.16014529, 0.4399212, 0.000001],
                                        [0.000001, 0.000001, 0.000001, 0.000001, 0.000001]]).cuda())
        
        # dinuc overfit on N.Crassa
        ncrassa_dist = torch.log(torch.tensor([[0.27720314, 0.20685077, 0.24993863, 0.26600747, 0.000001],
                                        [0.30233271, 0.21791942, 0.20990551, 0.26984236, 0.000001],
                                        [0.28934538, 0.19029556, 0.28043999, 0.23991907, 0.000001],
                                        [0.18967726, 0.21642376, 0.28722743, 0.30667156, 0.000001],
                                        [0.000001, 0.000001, 0.000001, 0.000001, 0.000001]]).cuda())
        
        # choose dinucleotide distribution
        self.di_dist = {'train':train_dist,
                            'scer':scer_dist,
                            'spombe':spombe_dist,
                            'ncrassa':ncrassa_dist}.get(dinuc_for,train_dist)


    def forward(self, x):
        # at each position to predict check previous output and give predicted second letter 
        # move by one

        # expect x to be (B,dim,L)

        out = torch.matmul(x.transpose(1,-1),self.di_dist).transpose(1,-1).cuda()
        out[1:] = out[:-1].clone()

        return out 
