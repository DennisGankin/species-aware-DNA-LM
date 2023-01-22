"""Sequence encoder classes"""

from src.datamodules.sequence_operations import *
import numpy as np

class RollingMasker():
    def __init__(
        self,
        mask_stride: int = 30,
        masker=np.zeros(5),
        frame: int = 0
        ) -> None:
        self.masker = masker
        self.mask_stride = mask_stride
        self.frame = frame
    
    def __call__(self, seq, motifs):
        return rolling_mask_test(
            seq,
            motifs,
            mask_stride = self.mask_stride,
            masker = self.masker,
            frame = self.frame
            )

class PlainOneHot():
    def __init__(
        self,
        frame: int = 0
        ) -> None:
        self.frame = frame
    def __call__(self, seq, motifs):
        return plain_one_hot(seq,motifs, frame=self.frame)
    

class RollingExoMasker():
    def __init__(
        self,
        mask_stride: int = 30,
        masker=np.zeros(5),
        frame: int = 0
        ) -> None:
        self.masker = masker
        self.mask_stride = mask_stride
        self.frame = frame
    
    def __call__(self, seq):
        return rolling_exo_mask_test(
            seq,
            mask_stride = self.mask_stride,
            masker = self.masker,
            frame = self.frame
            )

class MotifMasker():
    def __init__(
        self,
        seq_len: int = 300,
        total_len: int = 303,
        masker = np.zeros(5)
        ) -> None:
        self.seq_len = seq_len
        self.total_len = total_len
        self.masker = masker
    
    def __call__(self, seq, motifs):
        return motif_mask_encoding(
            seq,
            motifs,
            seq_len=self.seq_len,
            total_len=self.total_len,
            masker=self.masker)

class SequenceDataEncoder():
    """
    Encodes sequences for training
    Takes sequence as input and returns needed training data
    """
    def __init__(
        self,
        seq_len: int = 300,
        total_len: int = 305,
        mask_rate: float = 0.1,
        split_mask: bool = False,
        frame: int = 30,
        ):
        self.seq_len = seq_len
        self.total_len = total_len
        self.mask_rate = mask_rate
        self.split_mask = split_mask
        self.frame = frame
    
    def __call__(self, seq, motifs):
        return encode_and_pad(
            seq,
            seq_len=self.seq_len,
            total_len=self.total_len,
            mask_rate=self.mask_rate,
            split_mask=self.split_mask, 
            frame=self.frame)

class SequenceDataEncoderTest():
    """
    Encodes sequences for testing
    Takes sequence as input and returns needed testing data
    """
    def __init__(
        self,
        seq_len: int = 300,
        total_len: int = 305,
        mask_rate: float = 0.1,
        split_mask: bool = False,
        frame: int = 0
        ):
        self.seq_len = seq_len
        self.total_len = total_len
        self.mask_rate = mask_rate
        self.split_mask = split_mask
        self.frame = frame
    
    def __call__(self, seq, motifs):
        return encode_and_pad_test(
            seq,
            motifs,
            seq_len=self.seq_len,
            total_len=self.total_len,
            mask_rate=self.mask_rate,
            split_mask=self.split_mask, 
            frame=self.frame
            )

class TestTargetEncoder():
    """
    Class calling compute_target_seq
    takes sequence and mitifs and returns motif positions
    """
    def __init__(self, seq_len: int = 300, total_len: int = 305):
        self.seq_len = seq_len,
        self.total_len = total_len
    
    def __call__(self, seq, motifs):
        return compute_target_seq(seq, motifs, seq_len=self.seq_len, total_len=self.total_len)

class BatchSeqTransformer():
    """
    Class calling batched transform
    takes Dataframe with ["labels","lengths"] columns and transforms it to model input batch
    """

    def __init__(self, max_len: int = 300, mask_rate: float = 0.1):
        self.mask_rate = mask_rate
        self.max_len = max_len

    def __call__(self, batch_df):
        return batched_transform(batch_df, mask_rate = self.mask_rate, max_len = self.max_len)