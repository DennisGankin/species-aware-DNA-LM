import os
import sys
import yaml
import re
from os import path
from tqdm import tqdm

import pandas as pd
import numpy as np

import torch
from torch.nn.functional import cross_entropy, nll_loss, log_softmax

from Bio import AlignIO

from src.datamodules.sequence_utils import * #seq_to_labels


class WholeGenomeAlignment:
    
    def __init__(self, 
                 chromset, 
                 base_path,
                 target_prefix,
                 query_prefixes,
                 baseline_freqs):
        self.nt_dict = {"A":0,"C":1,"G":2,"T":3}
        self.baseline_freqs = baseline_freqs
        self.target_prefix = target_prefix
        self.query_prefixes = query_prefixes
        self.indices = {}
        # create alignment index objects for each chromosome maf file
        for chrom in chromset:
            idx = AlignIO.MafIO.MafIndex(path.join(base_path,"chr{}.mafindex".format(chrom)), 
                                         path.join(base_path,"chr{}.maf".format(chrom)), 
                                         "{}.chr{}".format(self.target_prefix,chrom))
            self.indices[chrom] = idx
    
    
    def get_alignment(self, chrom,
                      seq_start,
                      seq_end,
                      strand):
        alignment_idx = self.indices[chrom]
        multiple_alignment = alignment_idx.get_spliced(
                                    [seq_start-1], [seq_end],
                                    strand=strand)
        alignment = SeqAlignment(multiple_alignment, self)
        assert len(alignment.gapped_target_seq.replace("-","")) == seq_end - (seq_start - 1)
        return alignment
        
    
    def impute_for_motif(self,
                         chrom,
                         seq_start,
                         seq_end,
                         strand,
                         motif):
        prbs = []
        # get the region of interest from the alignment
        alignment = self.get_alignment(chrom,
                                       seq_start,
                                       seq_end,
                                       strand)
        # find matches
        gapped_matches = alignment.search_for_motif(motif)
        # impute for matches
        prbs = [alignment.impute_for_match(match) for match in gapped_matches]
        return prbs
    
    def impute_for_seq(self,
                       chrom,
                       seq_start,
                       seq_end,
                       strand):
        # get the region of interest from the alignment
        alignment = self.get_alignment(chrom,
                                       seq_start,
                                       seq_end,
                                       strand)
        return alignment.impute_for_all()
            
class SeqAlignment:
    
    def __init__(self, 
                 multiple_alignment, 
                 wg_alignment):
        self.aligned_seqs = {seqrec.name:str(seqrec.seq) for seqrec in multiple_alignment}
        self.wg_alignment = wg_alignment
        gapped_query_seqs = []
        for k in self.aligned_seqs:
            if k.startswith(self.wg_alignment.target_prefix):
                self.gapped_target_seq = self.aligned_seqs[k]
            elif any(k.startswith(x) for x in self.wg_alignment.query_prefixes):
                gapped_query_seqs.append(self.aligned_seqs[k])
        assert self.gapped_target_seq
        self.query_seqs = QuerySequences(gapped_query_seqs,
                                         self.wg_alignment.baseline_freqs,
                                         self.wg_alignment.nt_dict)
        self.nt_dict = self.wg_alignment.nt_dict
        
    def search_for_motif(self, motif):
        ungapped_target_seq = self.gapped_target_seq.replace("-","")
        # get ungapped start locations
        matches = [x for x in re.finditer(motif, ungapped_target_seq)]
        # gapped motif = ('-*').join(list(motif))
        # maybe re.finditer(r'(?=({}))'.format(motif),ungapped_target_seq)
        # convert to gapped start locations
        gapped_matches = []
        for match in matches:
            gapped_start_pos = self._jump_to_pos(0, 0, match.span()[0])
            gapped_end_pos = self._jump_to_pos(gapped_start_pos, match.span()[0], match.span()[1])
            assert match.group() == self.gapped_target_seq[gapped_start_pos:gapped_end_pos].replace("-","")
            gapped_match = GappedMatch(gapped_start_pos, 
                                       gapped_end_pos,
                                       match.group())
            gapped_matches.append(gapped_match)
        return gapped_matches
        
    def _jump_to_pos(self,
                     gapped_pos,
                     ungapped_pos,
                     target_pos):
        # deal with the edge case 
        # that alignment begins with a gap
        while self.gapped_target_seq[gapped_pos] == "-":
            gapped_pos += 1
        # move the pointer to the start of the match
        # ignoring gaps
        while ungapped_pos < target_pos:
            if self.gapped_target_seq[gapped_pos] != "-":
                ungapped_pos += 1
            gapped_pos += 1
        return gapped_pos
    
    def impute_for_match(self, gapped_match):
        prbs = []
        gapped_pos = gapped_match.start
        while gapped_pos < gapped_match.end:
            target_nt = self.gapped_target_seq[gapped_pos]
            # we do not impute anything for gaps/non-nt in the query
            if target_nt in self.nt_dict:
                prb_vector = self.query_seqs.impute_for_position(gapped_pos)
                prbs.append(prb_vector[self.nt_dict[target_nt]])
            gapped_pos += 1
        return np.mean(prbs)
    
    def impute_for_all(self):
        prbs = []
        gapped_pos = 0
        while gapped_pos < len(self.gapped_target_seq):
            target_nt = self.gapped_target_seq[gapped_pos]
            if target_nt != "-": # we do not impute anything for gaps in the query
                prbs.append(self.query_seqs.impute_for_position(gapped_pos))
            gapped_pos += 1
        return prbs
    
class QuerySequences:
    
    
    def __init__(self, 
                 gapped_query_seqs,
                 baseline_freqs,
                 nt_dict
                ):
        self.gapped_query_seqs = gapped_query_seqs
        self.baseline_freqs = baseline_freqs
        self.nt_dict = nt_dict
    
    def impute_for_position(self, pos):
        # impute the nucleotide at a particular position
        # from the alignment
        # if nothing aligns, impute using baseline
        if not self.gapped_query_seqs or len(self.gapped_query_seqs) == 0:
            return self.baseline_freqs
        # else impute using alignment
        prbs = []
        for seq in self.gapped_query_seqs:
            prb_vector = np.zeros(4)
            nt = self.nt_dict.get(seq[pos],None)
            if nt: # we only count non-gaps
                prb_vector[nt] = 1.0
                prbs.append(prb_vector)
        # if we had only gaps
        if len(prbs) == 0:
            return self.baseline_freqs
        # else we return the average over sequences
        return np.mean(prbs,axis=0)
    
class GappedMatch:
    
    def __init__(self, start, end, group):
        self.start = start
        self.end = end
        self.group = group

class AlignmentModel():

    def __init__(self, 
        halflife_df_path: str,
        alignment_base_path: str,
        ) -> None:

        self.halflife_df = pd.read_csv(halflife_df_path)

        self.wga = WholeGenomeAlignment(chromset=set(self.halflife_df["chr"]), 
            base_path=alignment_base_path,
            target_prefix="sacCer3",
            query_prefixes = ["sacCas","sacKlu"],#,"sacPar","sacMik","sacKud","sacBay"],
            baseline_freqs= np.array([0.25, 0.25, 0.25, 0.25])
            )     #np.array([0.33403754, 0.15997382, 0.14409694, 0.3618917]))


    def test(self):
        prbs = []

        counter = 0
        counter_seq = 0
        complete_string = "" 
        for _,row in self.halflife_df.iterrows():
            chrom = row["chr"]
            strand = -1 if row["strand"] == "-" else 1
            prbs += self.wga.impute_for_seq(chrom=chrom,
                                    seq_start=row["UTR3_start"],
                                    seq_end=row["UTR3_end"],
                                    strand=strand)
            counter += row["UTR3_end"] - row["UTR3_start"]
            counter_seq += len(row["UTR3_seq"])
            complete_string += row["UTR3_seq"]

        prbs = np.stack(prbs,axis=0)
        prbs = np.concatenate([prbs,np.zeros((prbs.shape[0],1))],axis=1)
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