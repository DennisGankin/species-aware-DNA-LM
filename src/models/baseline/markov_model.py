import numpy as np
from tqdm import tqdm
import itertools
import pickle
import pandas as pd

import torch

from torch.nn.functional import nll_loss

from Bio import AlignIO

from src.datamodules.sequence_utils import * #seq_to_labels


class KmerCount:
    
    @classmethod
    def init_from_file(cls, file):
        with open(file,'rb') as f:
            kmer_counts_dict = pickle.load(f)
        kmercount = cls(len(kmer_counts_dict))
        kmercount.kmer_counts_dict = kmer_counts_dict
        return kmercount
    
    def __init__(self, max_k, pseudocount = 0):
        self.max_k = max_k
        self.kmer_dict = {}
        self.kmer_counts_dict = {}
        for k in range(self.max_k+1):
            #print(k)
            kmers = {"".join(x):i for i,x in zip(range(4**k), itertools.product("ACGT",repeat=k))}
            self.kmer_dict[k] = kmers
            counts = np.zeros(4**k) + pseudocount # we pseudocount everything
            self.kmer_counts_dict[k] = counts
        
    def _update_on_seq(self, seq):
        i = 0
        while i < len(seq):
            k = 1
            while i < len(seq) - k + 1 and k <= self.max_k:
                kmer = seq[i:i+k]
                if "N" in kmer:
                    i += 1
                    break
                self.kmer_counts_dict[k][self.kmer_dict[k][kmer]] += 1
                k += 1
            i += 1
            
    def compute_counts(self, seq_vec):
        for seq in tqdm(seq_vec):
            self._update_on_seq(seq)
            
    def serialize_counts(self, file):
        with open(file,"wb") as f:
            pickle.dump(self.kmer_counts_dict, f)

class Markov:
    
    def serialize(self, file):
        assert np.nansum(self.markov_matrix) > 0
        np.save(file, self.markov_matrix)
        
    def load_matrix(self, file):
        markov_matrix = np.load(file)
        assert self.markov_matrix.shape == markov_matrix.shape()
        
    def impute_for_seq_vec(self, seq_vec, order=None):
        return np.concatenate([self.impute_for_seq(seq, order=order) for seq in seq_vec],axis=0) 

class MarkovChain(Markov):
    
    @classmethod
    def init_from_file(cls, file):
        markov_matrix = np.load(file)
        mkv = cls(KmerCount(markov_matrix.shape[0]))
        mkv.markov_matrix = markov_matrix
        return mkv
    
    def __init__(self, kmercount):
        self.max_k = kmercount.max_k
        self.kmer_dict = kmercount.kmer_dict
        self.kmer_counts_dict = kmercount.kmer_counts_dict
        self.markov_matrix = np.zeros((self.max_k,len(self.kmer_dict[self.max_k-1]),4))
    
    def compile_from_counts(self):
        for k in range(1,self.max_k+1):
            for kmer in self.kmer_dict[k]:
                state = [self.kmer_dict[k-1][kmer[:k-1]]]
                next_nt = [self.kmer_dict[1][kmer[k-1]]]
                counts = self.kmer_counts_dict[k][self.kmer_dict[k][kmer]]
                self.markov_matrix[k-1, state, next_nt] = counts
            # normalize rows
            if k == 1:
                self.markov_matrix[k-1,:,:] = self.markov_matrix[k-1,:,:]/np.sum(self.markov_matrix[k-1,:,:])
            else:
                self.markov_matrix[k-1,:,:] = self.markov_matrix[k-1,:,:]/np.sum(self.markov_matrix[k-1,:,:],axis=1)[:,np.newaxis]
        
    def impute_for_seq(self, seq, order=None):
        probs = []
        if not order and not order == 0:
            order = self.max_k - 1
        assert order <= self.max_k - 1
        i = 0
        j = 0
        while j < order and i < len(seq) - j:
            # for the first few nt, use a smaller order
            # e.g. first nt we impute with zero order etc
            state = [self.kmer_dict[j][seq[i:i+j]]]
            probs.append(self.markov_matrix[j][state])
            j += 1
        while i < len(seq) - order:
            state = [self.kmer_dict[order][seq[i:i+order]]]
            probs.append(self.markov_matrix[order][state])
            i += 1
        return np.concatenate(probs,axis=0)
    
class BiMarkov(Markov):
    
    @classmethod
    def init_from_file(cls, file):
        markov_matrix = np.load(file)
        #print(file)
        #print(markov_matrix.shape[0])
        mkv = cls(KmerCount(markov_matrix.shape[0]*2 + 1))
        mkv.markov_matrix = markov_matrix
        return mkv
    
    def __init__(self, kmercount):
        self.max_k = kmercount.max_k
        assert self.max_k >= 3
        assert self.max_k % 2 == 1
        self.kmer_dict = kmercount.kmer_dict
        self.kmer_counts_dict = kmercount.kmer_counts_dict
        self.markov_matrix = np.zeros((self.max_k//2 + 1,len(self.kmer_dict[self.max_k-1]),4))
    
    def compile_from_counts(self):
        for order in range(0,(self.max_k)//2 + 1):
            k = (order*2)+1
            for kmer in self.kmer_dict[k]:
                state = [self.kmer_dict[k-1][kmer[:order] + kmer[order+1:]]]
                middle_nt = [self.kmer_dict[1][kmer[order]]]
                counts = self.kmer_counts_dict[k][self.kmer_dict[k][kmer]]
                self.markov_matrix[order, state, middle_nt] = counts
            # normalize rows
            if k == 1:
                self.markov_matrix[order,:,:] = self.markov_matrix[order,:,:]/np.sum(self.markov_matrix[order,:,:])
            else:
                self.markov_matrix[order,:,:] = self.markov_matrix[order,:,:]/np.sum(self.markov_matrix[order,:,:],axis=1)[:,np.newaxis]
        
    def impute_for_seq(self, seq, order=None):
        probs = []
        if not order and not order == 0:
            order = self.max_k//2
        assert order <= self.max_k//2
        i = 0
        j = 0
        while j < order and i < len(seq) - j:
            kmer = seq[i-j:i] + seq[(i+1):(i+j+1)]
            state = [self.kmer_dict[j*2][kmer]]
            probs.append(self.markov_matrix[j][state])
            i += 1
            j += 1
        while i < len(seq) - order:
            kmer = seq[i-order:i] + seq[(i+1):(i+order+1)]
            state = [self.kmer_dict[order*2][kmer]]
            probs.append(self.markov_matrix[order][state])
            i += 1
        while j >= 0:
            while i < len(seq) - j:
                # for the last few nt, use a smaller order
                kmer = seq[i-j:i] + seq[(i+1):(i+j+1)]
                state = [self.kmer_dict[j*2][kmer]]
                probs.append(self.markov_matrix[j][state])
                i += 1
            j -= 1
        return np.concatenate(probs,axis=0)

class MarkovModel():

    def __init__(self, 
        halflife_df_path: str,
        markov_matrix_path: str,
        order: int,
        bidirectional: bool) -> None:

        self.halflife_df = pd.read_csv(halflife_df_path)
        
        if bidirectional:
            self.mkv = BiMarkov.init_from_file(markov_matrix_path)
        else:
            self.mkv = MarkovChain.init_from_file(markov_matrix_path)
        self.order = order
        

    def test(self):
        prbs = []
        complete_string = ""

        for _,row in self.halflife_df.iterrows():
            prbs.append(self.mkv.impute_for_seq(row["UTR3_seq"], order=self.order))
            complete_string += row["UTR3_seq"]

        prbs = np.concatenate(prbs,axis=0)
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
