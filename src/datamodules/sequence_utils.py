"""Utilities for sequence operations"""

import numpy as np

# global mapping of nucleotide to classes
mapping = dict(zip(b"ACGTN", range(5)))
mapping.update(dict(zip("ACGTN", range(5))))

def seq_to_labels(seq):
    """
    Computes class labels from sequence
    """
    # allow string and binary string #put this to class
    seq2 = []
    for i in seq:
        try:
            seq2.append(mapping[i])
        except:
            seq2.append(mapping["N"])
    return seq2

def seq_to_labels_padded(seq, max_len, pad=-100):
    """
    Computes class labels from sequence and
    Pads sequence to max_len. Pad on the right
    """
    label_vec = [-100] * max_len
    
    for i in range(len(seq[:max_len])):
        residue = seq[i]
        label_vec[i] = mapping.get(residue, mapping["N"])
    return label_vec

def one_hot(labels):
    """
    Computes one hot vector as np.array from label vector or list
    """
    return np.eye(5)[labels]

def one_hot_encode(seq):
    """
    Computes class labels and one hot encoding of DNA sequence passed as string or binary string
    """
    # allow string and binary string
    mapping = dict(zip(b"ACGTN", range(5)))
    mapping.update(dict(zip("ACGTN", range(5))))
    seq2 = []
    for i in seq:
        try:
            seq2.append(mapping[i])
        except:
            seq2.append(mapping["N"])
    # return as sequence of class labels and as one hot encoded sequence
    seq2 = np.array(seq2)
    return seq2, np.eye(5)[seq2]


def pad_all(seq_one_hot, seq_labels, masks, total_len):
    """
    Computes padding for sequence vectors and the mask
    """
    # length of padding to each side
    pad_len = (total_len - len(seq_labels))/2
    
    #pad with zeros at each side, if uneven one pad more after than before
    before = int(np.floor(pad_len))
    after = int(np.ceil(pad_len))
    seq_one_hot = pad_one_hot(seq_one_hot, before, after)
    # pad sequence labels with ignore index 
    seq_labels = pad_1d(seq_labels, before, after, pad=-100.0)
    padded_masks = []
    for mask in masks:
        padded_masks.append(pad_1d(mask, before, after))
    
    return seq_one_hot, seq_labels, padded_masks  

def pad_one_hot(seq, before, after):
    """
    Pad with [0,0,0,0,0] arrays
    """
    return np.concatenate((np.zeros((before,5)),seq,np.zeros((after,5))),axis=0)

def pad_1d(mask, before, after, pad=0.0):
    """
    Pad with zeros
    """
    return np.concatenate((np.full((before),pad),mask,np.full((after),pad)),axis=0)
    