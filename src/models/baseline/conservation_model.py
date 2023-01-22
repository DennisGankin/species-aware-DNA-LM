import pyBigWig
import pandas as pd
import numpy as np

import torch

from torch.nn.functional import nll_loss

from Bio import AlignIO

from src.datamodules.sequence_utils import * #seq_to_labels

class ConservationTrack:
    
    def __init__(self, path):
        self.bb = pyBigWig.open(path)
        
    def get_conservation(self, chrom, start, end, strand):
        if chrom == "Mito":
            return np.zeros(end - start)
        vals = self.bb.values("chr"+chrom, start, end)
        assert len(vals) == end - start
        vals = np.nan_to_num(vals)
        if strand == "-":
            vals = vals[::-1]
        return [vals]

class ConservationModel:
    
    def __init__(self, 
        halflife_df_path: str,
        conservation_track_path: str) -> None:

        self.halflife_df = pd.read_csv(halflife_df_path)
        
        self.conservation = ConservationTrack(conservation_track_path)
        

    def test(self):
        prbs = []
        complete_string = ""


        for _,row in self.halflife_df.iterrows():
            chrom = row["chr"]
            prbs += self.conservation.get_conservation(chrom=chrom,
                                          start=row["UTR3_start"]-1,
                                          end=row["UTR3_end"],
                                          strand=row["strand"])
            complete_string += row["UTR3_seq"]
        
        prbs = np.concatenate(prbs,axis=0)
        prbs = prbs.repeat(5).reshape(-1,5)
        prbs = torch.from_numpy(prbs)

        # get targets
        targets = torch.tensor(seq_to_labels(complete_string))

        # compute cross entropy, it's already as probability so just nll
        ce = nll_loss(prbs,targets, reduction="none") #cross_entropy(prbs, targets)

        #print(ce)

        # save
        torch.save(prbs, "masked_logits.pt") # no logits, so use prbs
        torch.save(torch.argmax(prbs, dim=1), "masked_preds.pt")
        torch.save(prbs, "prbs.pt")
        torch.save(ce,"ce.pt")

        # save targets
        torch.save(targets, "masked_targets.pt")

        # save rest as placeholders (zeros of same length)
        torch.save(torch.zeros(len(prbs)),"masked_motifs.pt")