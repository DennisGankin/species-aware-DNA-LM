"""
Sequence operations for sequence encoders
"""

import numpy as np
import torch
import re
from src import utils
from src.datamodules.sequence_utils import *

log = utils.get_logger(__name__)

# global mapping of nucleotide to classes
mapping = dict(zip(b"ACGTN", range(5)))
mapping.update(dict(zip("ACGTN", range(5))))

def random_masking(
                seq_vector,
                mask_rate = 0.1,
                masker = np.zeros(5),
                split_mask = False,
                frame=0):
    """
    Masks a given sequence vector at random places, returns masked vector. 
    Masks exactly the fraction of positions as given by masking_rate 
    (80% with mask token, 10% random, 10% keep)
    if frame given, we do not mask the edges
    """
    
    #print("Config")
    #print (mask_rate)
    #print(split_mask)
    #print(frame)

    mask = np.zeros(len(seq_vector), dtype=int)

    # if sequence to small
    if 2*frame >= len(seq_vector):
        #print("too small")
        return seq_vector, mask

    # framed sequence length
    seq_len = len(seq_vector) - 2 * frame 
    framed_mask = np.zeros(seq_len, dtype=int)

    if split_mask:
        # mask 80% with zeros
        idx_masked = int(np.rint((mask_rate*0.8)*seq_len))
        # mask 10% with radom
        idx_random = idx_masked + max(int(np.rint(mask_rate*0.1*seq_len)), 1)
        # mask 10% with same
        idx_keep = idx_random + int(np.rint((mask_rate*0.1)*seq_len))
        # number of places to mask
        #num_masked = int(mask_rate*len(seq_vector))
        # add ones to the array and shuffle randomly
        framed_mask[:idx_masked] = 1
        framed_mask[idx_masked:idx_random] = 2
        framed_mask[idx_random:idx_keep] = 3
    else:
        idx_masked = int(np.rint((mask_rate)*seq_len))
        framed_mask[:idx_masked] = 1

    np.random.shuffle(framed_mask)
    if frame!=0:
        mask[frame:-frame] = framed_mask
    else:
        mask = framed_mask

    # mask with zeros
    seq_vector[mask==1] = masker

    # mask with random
    #if idx_random - idx_masked != 0:
    if split_mask:
        tmp_masker = [one_hot(mapping[np.random.choice(["A","C","T","G"])]) for i in range(idx_random-idx_masked)]
        seq_vector[mask==2] = tmp_masker

    # make everything 1
    mask[mask!=0] = 1

    #assert sum(mask)

    return seq_vector, mask

def encode_and_pad(
                seq,
                seq_len=300,
                total_len=303,
                mask_rate=0.1,
                split_mask=False,
                frame=0,
                ):
    """
    Creates encoded, masked and padded
    input sequence (one hot encoded) 
    target sequence (as class labels)
    used mask
    Some sequences seem to be longer (303) due to inconsistencies in start codons
    Do random masking based on mask rate
    """

    # truncate all to same length
    seq = seq[:seq_len]
    # one hot encode
    seq_labels, seq_one_hot = one_hot_encode(seq)

    # mask the sequence, masked sequence remains one hot encoded
    masked_seq, mask = random_masking(seq_one_hot,
                            mask_rate=mask_rate,
                            split_mask=split_mask,
                            frame=frame)
    
    # pad sequence and masked sequence and mask
    masked_seq, seq_labels, masks = pad_all(masked_seq, seq_labels, [mask], total_len)
    mask = masks[0]

    # return class labels as targets instead of one hot
    # target_sequence = torch.argmax(torch.from_numpy(seq_one_hot.transpose()).float(), dim=0)
    # only keep target nucleotides that where masked in input, to compute accurracy 
    seq_labels_masked = seq_labels.copy()
    seq_labels_masked[mask==0] = -100.0

    # return masked vector, normal vector and mask
    # make sure x has type float and labels type long 

    x = torch.from_numpy(masked_seq.transpose()).float()
    y = torch.from_numpy(seq_labels.transpose()).long()
    y_masked = torch.from_numpy(seq_labels_masked.transpose()).long()

    return x, y_masked, y, mask, []

def encode_and_pad_test(
                    seq,
                    motifs,
                    seq_len=300,
                    total_len=303,
                    mask_rate=0.1,
                    split_mask=False,
                    frame=0,
                    ):
    """
    Create all needed sequences for testing
    """
    # truncate all to same length
    seq = seq[:seq_len]
    # one hot encode
    seq_labels, seq_one_hot = one_hot_encode(seq)

    # do not mask for testing 
    # create same legth zeros
    motif_target_seq = np.zeros(len(seq))

    # mask the sequence, masked sequence remains one hot encoded
    masked_seq, mask = random_masking(
                                seq_one_hot.copy(), 
                                mask_rate=mask_rate,
                                split_mask=split_mask,
                                frame=frame)

    # for each motif
    for motif in motifs.keys():
        # find all occurances
        for match in re.finditer(motif, seq):
            # set found positions to motif id
            motif_target_seq[match.start():match.end()] = motifs[motif]

    # pad sequence and masked sequence and motif targets
    #seq_one_hot, seq_labels, motif_mask = pad_all(seq_one_hot, seq_labels, target_seq, total_len)
    seq_one_hot, seq_labels, masks = pad_all(masked_seq, seq_labels, (motif_target_seq, mask), total_len)
    motif_mask = masks[0]
    mask = masks[1]

    seq_labels_masked = seq_labels.copy()
    seq_labels_masked[mask==0] = -100.0

    # return masked vector, normal vector and mask
    # make sure x has type float and labels type long 
    x = torch.from_numpy(seq_one_hot.transpose()).float()
    y = torch.from_numpy(seq_labels.transpose()).long()
    y_masked = torch.from_numpy(seq_labels_masked.transpose()).long()

    return x, y_masked, y, mask, motif_mask

def rolling_mask(seq_one_hot, stride, masker=np.zeros(5), frame=0):
    """
    Creates a batch with all bases masked at some point in rolling mask fashion
    [1,0,0,1,0,0]
    [0,1,0,0,1,0]
    [0,0,1,0,0,1]
    """
    # make sure frame not too big 
    if 2 * frame >= len(seq_one_hot):
        frame = 0
    # adjust sequence length to frame (no prediction at edges)
    seq_len = len(seq_one_hot)# - 2 * frame

    # for all possible starts (between 0 and stride), get indices of masked positions
    mask_idx = [range(i+frame,seq_len-frame,stride) for i in range(stride)]
    batch_size = len(mask_idx)
     
    # create masks and masked_sequences
    stacked_seqs = np.full((batch_size,seq_one_hot.shape[0],seq_one_hot.shape[1]),seq_one_hot)
    masks = np.zeros((batch_size,len(seq_one_hot)))
    for i in range(batch_size):
        masks[i,mask_idx[i]] = 1
        stacked_seqs[i,mask_idx[i]] = masker
    
    return stacked_seqs, masks
    

def plain_one_hot(seq, motifs, frame=0):

    # encode one hot 
    seq_labels, seq_one_hot = one_hot_encode(seq)

    # find motifs
    motif_mask = np.zeros(len(seq))
    # for each motif
    for motif in motifs.keys():
        # find all occurances
        for match in re.finditer(motif, seq):
            # set found positions to motif id
            motif_mask[match.start():match.end()] = motifs[motif]

    x = torch.from_numpy(seq_one_hot.transpose()).float().unsqueeze(0)
    y = torch.from_numpy(seq_labels.transpose()).long().unsqueeze(0)
    y_masked = y
    mask = torch.ones(len(y)).unsqueeze(0)
    motif_mask =  torch.from_numpy(motif_mask).unsqueeze(0)

    return x, y_masked, y, mask, motif_mask


def rolling_mask_test(seq, motifs, mask_stride=30, masker=np.zeros(5), frame=0):
    """
    Create all needed sequences for testing
    """
    # truncate all to same length
    #seq = seq[:seq_len]
    # one hot encode
    seq_labels, seq_one_hot = one_hot_encode(seq)

    # do not mask for testing 
    # create same legth zeros
    motif_mask = np.zeros(len(seq))

    # pad first, remember before and pass it to rolling mask

    # for each motif
    for motif in motifs.keys():
        # find all occurances
        for match in re.finditer(motif, seq):
            # set found positions to motif id
            motif_mask[match.start():match.end()] = motifs[motif]
    
    #test = np.sum(target_seq)
    #log.info(test)

    # pad sequence and masked sequence and motif targets
    #seq_one_hot, seq_labels, motif_mask = pad_all(seq_one_hot, seq_labels, target_seq, total_len)
    #seq_one_hot, seq_labels, masks = pad_all(seq_one_hot, seq_labels, [target_seq], total_len)
    #motif_mask = masks[0]

    # length of padding to each side
    #pad_len = (total_len - len(seq_labels))/2
    
    #pad with zeros at each side, if uneven one pad more after than before
    #before = int(np.floor(pad_len))

    # mask the sequence, masked sequence remains one hot encoded
    x_batch, mask_batch = rolling_mask(seq_one_hot, mask_stride, masker=masker, frame=frame)

    batch_size = len(mask_batch)

    # create batches
    motif_mask_batch = np.full((batch_size,len(seq)), motif_mask)
    y_batch = np.full((batch_size,len(seq)), seq_labels)
    y_masked_batch = y_batch.copy()
    y_masked_batch[mask_batch == 0] = -100.0 # apply mask

    # convert to torch tensors
    x_batch = torch.from_numpy(x_batch.transpose((0,2,1))).float() 
    y_masked_batch = torch.from_numpy(y_masked_batch).long()
    y_batch = torch.from_numpy(y_batch).long()
    motif_mask_batch = torch.from_numpy(motif_mask_batch)
    mask_batch = torch.from_numpy(mask_batch)

    assert len(seq) == torch.sum(mask_batch), str(len(seq)) + " vs. "+ str(torch.sum(mask_batch)) + " --- " + seq

    return x_batch, y_masked_batch, y_batch, mask_batch, motif_mask_batch


def rolling_exo_mask_test(seq_data, mask_stride=30, masker=np.zeros(5), frame=0):
    """
    create rolling mask on sequences with exo peak motifs
    motif positions already passed inside pandas Series
    """
    # assess Series columns
    seq = seq_data.seq
    motif_mask = seq_data.motif_indication

    # one hot encode
    seq_labels, seq_one_hot = one_hot_encode(seq)

    # mask the sequence, masked sequence remains one hot encoded
    x_batch, mask_batch = rolling_mask(seq_one_hot, mask_stride, masker=masker, frame=frame)

    batch_size = len(mask_batch)

    # create batches
    motif_mask_batch = np.full((batch_size,len(seq)), motif_mask)
    y_batch = np.full((batch_size,len(seq)), seq_labels)
    y_masked_batch = y_batch.copy()
    y_masked_batch[mask_batch == 0] = -100.0 # apply mask

    # convert to torch tensors
    x_batch = torch.from_numpy(x_batch.transpose((0,2,1))).float() 
    y_masked_batch = torch.from_numpy(y_masked_batch).long()
    y_batch = torch.from_numpy(y_batch).long()
    motif_mask_batch = torch.from_numpy(motif_mask_batch)
    mask_batch = torch.from_numpy(mask_batch)

    return x_batch, y_masked_batch, y_batch, mask_batch, motif_mask_batch


def motif_mask_encoding(seq,motifs,seq_len=300,total_len=303,masker=np.zeros(5)):
    """
    Creates masked batches of one sequence with different positions of the motifs masked each time
    """
    # fill these later
    x_batch = []
    y_masked_batch = []
    mask_batch = []

    log.info("Go")
    # truncate all to same length
    seq = seq[:seq_len]
    # one hot encode
    seq_labels, seq_one_hot = one_hot_encode(seq)

    # create same legth zeros for motif mask
    target_seq = np.zeros(len(seq))

    motif_idx = []
    motif_idx_transposed = []

    # for each motif get ids
    for motif in motifs.keys():
        # find all occurances
        log.info("search")
        for match in re.finditer(motif, seq):
            log.info("found")
            # set found positions to motif id
            target_seq[match.start():match.end()] = motifs[motif]
            # append all indies of the motif to index list
            idx = [i for i in range(match.start(),match.end())]
            motif_idx.append([i for i in range(match.start(),match.end())])

            for i in range(len(idx)):
                ## go through list [[1,9],[2,10],[3]]
                # check if there is an id list at position, if not create new one
                if i < len(motif_idx_transposed):
                    motif_idx_transposed[i] += [idx[i]]
                else:
                    motif_idx_transposed.append([idx[i]])

    log.info("len ids" + str(len(motif_idx_transposed)))

    # get slices, and use them to mask sequence
    for motif_slice in motif_idx_transposed: 
        # mask one hot vektor there
        one_hot_tmp = seq_one_hot.copy()

        # mask at these ids
        one_hot_tmp[motif_slice] = masker

        log.info("Onehot shape" + str(one_hot_tmp.shape))
        # create mask
        mask = np.zeros(total_len)
        mask[motif_slice] = 1

        # mask seq labels
        # mask unimprotant labels
        seq_labels_masked = seq_labels.copy()
        seq_labels_masked[mask==0] = -100.0

        # pad one hot, masked sequence labels, mask

        # pad one hot
        # pad mask
        one_hot_tmp, seq_labels_masked, masks = pad_all(one_hot_tmp, seq_labels_masked, (mask), total_len)
        mask = masks[0]

        # add to batch 
        x_batch.append(one_hot_tmp) #torch.from_numpy(one_hot_tmp.transpose()).float()
        log.info("appended, len" + str(len(x_batch)))
        assert 1==2, "waht"
        y_masked_batch.append(seq_labels_masked) # torch.from_numpy(seq_labels_masked.transpose()).long()
        mask_batch.append(mask)
        
    # pad sequence labels
    # pad motif mask
    # pad motif mask
    # length of padding to each side
    pad_len = (total_len - len(seq_labels))/2   
    #pad with zeros at each side, if uneven one pad more after than before
    before = int(np.floor(pad_len))
    after = int(np.ceil(pad_len))

    motif_mask = pad_1d(target_seq, before, after)
    seq_labels = pad_1d(seq_labels, before, after, pad = -100.0)

    # create batches
    # stack up motif masks x_batch times
    motif_mask_batch = np.full((len(x_batch),len(motif_mask)), motif_mask)
    # stack up sequence labels len(x_batch) times
    y_batch = np.full((len(x_batch),len(seq_labels)), seq_labels)

    # transform from list of arrays to matrices
    x_batch = np.array(x_batch) 
    y_masked_batch = np.array(y_masked_batch)
    mask_batch = np.array(mask_batch)

    # convert to torch tensors
    x_batch = torch.from_numpy(x_batch).float() 
    y_masked_batch = torch.from_numpy(y_masked_batch).long()
    y_batch = torch.from_numpy(y_batch).long()

    # return masked one hot batch, masked targets (labels and -100 else), all labels, mask, 
    #print (x_batch)
    return x_batch, 1
    return x_batch, y_masked_batch, y_batch, mask_batch, motif_mask_batch


    #create test set with sequences that include motifs only


def compute_target_seq(seq, motifs, seq_len = 300, total_len = 305):
    """
    Creates vector that indicates motif positions in the sequence. 
    e.g: [0,0,0,1,1,1,1,0,0,0,0,3,3,3,3,3,0,0,0] if there are motifs 1 and 3 in the sequence
    Vector is padded to match the length of the sequence embeddings.
    """
    # truncate
    seq = seq[:seq_len]
    
    # create same legth zeros
    target_seq = np.zeros(len(seq))
    
    # for each motif
    for motif in motifs.keys():
        # find all occurances
        for match in re.finditer(motif, seq):
            # set found positions to motif id
            target_seq[match.start():match.end()] = motifs[motif]
    
    # add padding to match sequence vectors
    # length of padding to each side
    pad_len = (total_len - len(seq))/2
    
    #pad with zeros at each side, if uneven one pad more after than before
    before = int(np.floor(pad_len))
    after = int(np.ceil(pad_len))
    return np.concatenate((np.zeros(before),target_seq,np.zeros(after)),axis=0)


#### Batched functions

def random_batch_mask(batch, mask_rate, pad=[0,0,0,0,0]):
    """
    Creates random mask for given batch with dimensions [N,L] and masks one hot tensor accordingly
    mask_rate defines the rate of masked elements in the batch
    N - batch size 
    L - sequence length
    """
    mask = torch.zeros((batch.shape[0],batch.shape[1]))
    # mask with a certain mask rate
    mask[:,:int(np.rint(mask_rate*batch.shape[1]))] = 1
    # shuffle the elements completely random by assigning new indices
    # see https://discuss.pytorch.org/t/shuffling-a-tensor/25422/4
    idx = torch.randperm(mask.nelement())
    mask = mask.view(-1)[idx].view(mask.size())
    
    # set all one_hot vectors to be masked to [0,0,0,0,0] 
    batch[mask==1] = pad
    
    return batch, mask

def one_hot_(x):
    """
    Computes one hot vector as np.array from label vector or list
    """
    assert x!=1, "Nah" # no idea why but i get an error without this line (mybe some swifter thing?)
    return np.eye(5)[x]

def one_hot_and_pad(seq, length, max_len):
    if length == max_len:
        return one_hot_(seq)
    else:
        # padding with [0,0,0,0,0] to the right
        return np.concatenate([one_hot_(seq[:length]),np.zeros((max_len-length,5))], axis=0)

def batched_transform(batch, mask_rate=0.1, max_len=300):
    """
    One hot encodes and masks whole batch.
    batch passed as df with labels and lengths as columns
    returns input, target and mask as batch for the network
    """

    # when loading data for training
    # one hot encode only the length and add padding
    x = batch[["labels","lengths"]].swifter.progress_bar(False).apply(
        lambda row: one_hot_and_pad(row.labels, row.lengths, max_len),
        axis=1
        )
    x = np.stack(x)

    # create masking on whole batch
    x, mask = random_batch_mask(x,mask_rate)

    # set all non-masked labels to -100, to ignore them when computing accuracy
    y = np.stack(batch["labels"])
    y[mask==0] = -100

    # set all padding parts to zero/-100 again in the mask so that accurcy not computed on padding (remove masking there)
    # padding is -100 already 
    mask[y==-100] = 0

    # transpose last two dimensions and save as tensor
    x = torch.from_numpy(x.transpose(0,2,1)).float()
    y = torch.from_numpy(y).long()

    return x, y, mask, []
