import h5py
import pyreadr
import pandas as pd
import swifter
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST

from src import utils
from .sequence_utils import seq_to_labels_padded
from .sequence_encoders import batched_transform
from . import exo_motifs, spec_dict

from .motifs import pombe_utr3_motif_handler


log = utils.get_logger(__name__)

class HDF5ContigDataset(Dataset):
    """
    Dataset of sequences saved in HDF5 files. Created as contiguous dataset.
    """

    def __init__(
        self,
        hdf5_datatable,
        contig_size
        ) -> None:
        super().__init__()

        self.hdf5_datatable = hdf5_datatable
        self.contig_size = contig_size
        self.len = len(self.hdf5_datatable)
        self.max_index = self.len - self.contig_size

    def __len__(self) -> int:
        return self.len()

    def __getitem__(self, index):

        if index > self.len or index < 0:
            raise ValueError("hd5 index out of bounds") 

        return 


class HDF5Dataset(Dataset):
    """
    Dataset of sequences saved in HDF5 files.
    Accessing hd5 file for each sequence.
    """

    def __init__(
        self,
        hdf5_file_path,
        ) -> None:
        super().__init__()

        # open hd5 file 
        self.hdf5_file = h5py.File(hdf5_file_path,"r")
        self.hdf5_datatable = self.hdf5_file["sequences"]
        self.len = len(self.hdf5_datatable)

    def __len__(self) -> int:
        return self.len()

    def __getitem__(self, index):

        return self.hdf5_datatable[index]

#class HDF5RAMDataset(Dataset):

class CSVBatchedDataset(Dataset):
    """
    Dataset read at once from CSV file.
    Make sure enough GB is allocated.
    """
    def __init__(
        self,
        csv_file_path,
        train: bool = True,
        transform = None,
        target_transform = None,
        batch_size: int = 128,
        max_seq_len: int = 300
        ) -> None:
        super().__init__()

        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.batch_size = batch_size
        self.max_len = transform.max_len

        # maybe put this somewhere else and make it configurable
        # motifs in the dataset
        self.target_motifs = {"TGTAAATA":1, "TGCAT":2, "ATATTC":3, "TTTTTTA":4}

        # load from csv file 
        log.info("Loading data from CSV file at " + csv_file_path)
        df = pd.read_csv(csv_file_path)

        # only take rows where sequence exists
        df = df[df['three_prime_region'].notnull()]

        # reset the index bc we removed NaN values
        data = df['three_prime_region'].reset_index(drop=True)

        # truncate sequences if max_len specified
        if self.max_len is not None:
            data = data.swifter.apply(lambda seq: seq[:self.max_len])
        # get sequence lengths and max in case it's None
        lengths = data.map(len)
        self.max_len = max(lengths)
        self.transform.max_len = self.max_len

        # compute labels and pad at the end to max_len
        # save as sequence of class labels "ACTA" -> [0,1,3,1]
        # swifter speeds up pandas apply
        labels = data.swifter.apply(lambda seq: seq_to_labels_padded(seq, self.max_len))

        # concatenate to df
        self.data_df = pd.concat([data, labels, lengths], axis=1)
        self.data_df.columns=["seqs","labels","lengths"]

        # compute length considering batches
        # not dropping the last batch 
        self.length = int(np.ceil(len(self.data_df)/self.batch_size))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index):

        batch = self.data_df[index*self.batch_size:(index+1)*self.batch_size]
        return self.transform(batch)
        #return batched_transform(batch)

class SeqDataset(Dataset):
    """
    Abstract
    """
    def __init__(
        self,
        csv_file_path: str = "" ,
        train: bool = True,
        batch_size: int = 128,
        transform = None,
        target_transform = None,
        seq_position = "three_prime_region",
        ) -> None:
        super().__init__()


class CSVDataset(SeqDataset):
    """
    Dataset read at once from CSV file.
    Make sure enough GB is allocated.
    """

    def __init__(
        self,
        csv_file_path,
        train = True,
        batch_size = 128,
        transform = None,
        seq_position = "three_prime_region",
        ) -> None:
        super().__init__()

        self.train = train
        self.transform = transform

        # maybe put this somewhere else and make it configurable
        # motifs in the dataset
        self.target_motifs = {"TGTAAATA":1, "TGCAT":2, "ATATTC":3, "TTTTTTA":4}
        #self.target_motifs = pombe_utr3_motif_handler.dict

        # if rds file load df accordingly
        if csv_file_path[-4:] == ".rds":
            # load from rsd file 
            log.info("Loading data from rds file at " + csv_file_path)
            df = pyreadr.read_r(csv_file_path)[None]
        else:
            # load from csv file 
            log.info("Loading data from CSV file at " + csv_file_path)
            df = pd.read_csv(csv_file_path)

        # only take rows where sequence exists
        df = df[df[seq_position].notnull()].reset_index(drop=True)
 
        # read necessary information and onehot encode sequences
        # could make that faster with pandas speedup
        # self.data = np.dstack(list(df['three_prime_region'].apply(encode_and_pad)))
        # This would take up 50GB, so just save sequences in RAM
        self.data = list(df[seq_position])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):

        if self.transform is not None:
            # transform the sequences to masked and non masked vectors
            masked_sequence, target_labels_masked, target_labels, mask, motif_mask = self.transform(self.data[index], self.target_motifs)

        return masked_sequence, target_labels_masked, target_labels, mask, motif_mask

class ExoPeakDataset(SeqDataset):
    """
    Dataset with motif indications from exo peaks
    """

    def __init__(
        self,
        csv_file_path,
        train = False,
        batch_size = 128,
        transform = None,
        seq_position = "five_prime_region",
        ) -> None:
        super().__init__()

        self.train = train
        self.transform = transform

        # maybe put this somewhere else and make it configurable
        # motifs in the dataset
        self.target_motifs = exo_motifs

        # if rds file load df accordingly
        if csv_file_path[-4:] == ".rds":
            # load from rsd file 
            log.info("Loading data from rds file at " + csv_file_path)
            df = pyreadr.read_r(csv_file_path)[None]
        else:
            # load from csv file 
            log.info("Loading data from CSV file at " + csv_file_path)
            df = pd.read_csv(csv_file_path)

        # only take rows where sequence exists
        df = df[df[seq_position].notnull()].reset_index(drop=True)

        # rename sequence column (from five_prime_region to seq)
        df.rename(columns = {seq_position:'seq'}, inplace = True)

        #numpy array saved as string
        df.motif_indication = df.motif_indication_pwm.swifter.apply(lambda x: np.fromstring(x[1:-1], sep=' '))
 
        # read necessary information and onehot encode sequences
        # could make that faster with pandas speedup
        # self.data = np.dstack(list(df['three_prime_region'].apply(encode_and_pad)))
        # This would take up 50GB, so just save sequences in RAM
        self.data = df
        

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):

        if self.transform is not None:
            # transform the sequences to masked and non masked vectors
            masked_sequence, target_labels_masked, target_labels, mask, motif_mask = self.transform(self.data.iloc[index])

        return masked_sequence, target_labels_masked, target_labels, mask, motif_mask


class CSVSpecDataset(SeqDataset):
    """
    Dataset read at once from CSV file.
    Make sure enough GB is allocated.
    """

    def __init__(
        self,
        csv_file_path,
        train = True,
        batch_size = 128,
        transform = None,
        seq_position = "three_prime_region",
        ) -> None:
        super().__init__()

        self.train = train
        self.transform = transform

        # maybe put this somewhere else and make it configurable
        # motifs in the dataset
        #self.target_motifs = {"TGTAAATA":1, "TGCAT":2, "ATATTC":3, "TTTTTTA":4}

        #pombe:
        self.target_motifs = pombe_utr3_motif_handler.dict


        # if rds file load df accordingly
        if csv_file_path[-4:] == ".rds":
            # load from rsd file 
            log.info("Loading data from rds file at " + csv_file_path)
            df = pyreadr.read_r(csv_file_path)[None]
        else:
            # load from csv file 
            log.info("Loading data from CSV file at " + csv_file_path)
            df = pd.read_csv(csv_file_path)

        # only take rows where sequence exists
        df = df[df[seq_position].notnull()].reset_index(drop=True)
 
        # read necessary information and onehot encode sequences
        # could make that faster with pandas speedup
        # self.data = np.dstack(list(df['three_prime_region'].apply(encode_and_pad)))
        # This would take up 50GB, so just save sequences in RAM
        self.data = list(df[seq_position])

        self.species_labels = [spec_dict[s] for s in list(df["species"])]
            

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):

        if self.transform is not None:
            # transform the sequences to masked and non masked vectors
            masked_seq, target_labels_masked, target_labels, mask, motif_mask = self.transform(self.data[index], self.target_motifs)
            spec_label = self.species_labels[index] #torch.Tensor(self.species_labels[index]).float()

            masked_sequence = (masked_seq, spec_label)

        """return {"masked_sequence":masked_sequence,
                "target_labels_masked":target_labels_masked,
                "target_labels":target_labels, 
                "mask": mask,
                "motif_mask": motif_mask,
                "species_labels": spec_label}
        """
        return masked_sequence, target_labels_masked, target_labels, mask, motif_mask



class CSVSpecTestDataset(SeqDataset):
    """
    Dataset read at once from CSV file.
    Make sure enough GB is allocated.
    """

    def __init__(
        self,
        csv_file_path,
        train = True,
        batch_size = 128,
        transform = None,
        seq_position = "three_prime_region",
        species_test_label = "saccharomyces_cerevisiae"
        ) -> None:
        super().__init__()

        self.train = train
        self.transform = transform

        # maybe put this somewhere else and make it configurable
        # motifs in the dataset
        #self.target_motifs = {"TGTAAATA":1, "TGCAT":2, "ATATTC":3, "TTTTTTA":4}

        #pombe:
        self.target_motifs = pombe_utr3_motif_handler.dict


        # if rds file load df accordingly
        if csv_file_path[-4:] == ".rds":
            # load from rsd file 
            log.info("Loading data from rds file at " + csv_file_path)
            df = pyreadr.read_r(csv_file_path)[None]
        else:
            # load from csv file 
            log.info("Loading data from CSV file at " + csv_file_path)
            df = pd.read_csv(csv_file_path)

        # only take rows where sequence exists
        df = df[df[seq_position].notnull()].reset_index(drop=True)
 
        # read necessary information and onehot encode sequences
        # could make that faster with pandas speedup
        # self.data = np.dstack(list(df['three_prime_region'].apply(encode_and_pad)))
        # This would take up 50GB, so just save sequences in RAM
        self.data = list(df[seq_position])
        if "species" in df.columns:
            self.species_labels = [spec_dict[s] for s in list(df["species"])]
        else: 
            log.info("Set label manually for testing")
            #self.species_labels = [spec_dict['saccharomyces_cerevisiae'] for s in self.data]
            # pombe 'schizosaccharomyces_pombe'
            #self.species_labels = [spec_dict['schizosaccharomyces_pombe'] for s in self.data]
            # self.species_labels = [spec_dict['amanita_muscaria'] for s in self.data]
            #self.species_labels = [spec_dict['candida_glabrata'] for s in self.data]
            #self.species_labels = [spec_dict['neurospora_crassa'] for s in self.data]
            self.species_labels = [spec_dict[species_test_label] for s in self.data]
            

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):

        if self.transform is not None:
            # transform the sequences to masked and non masked vectors
            masked_seq, target_labels_masked, target_labels, mask, motif_mask = self.transform(self.data[index], self.target_motifs)
            spec_label = self.species_labels[index] #torch.Tensor(self.species_labels[index]).float()

            # hot fix
            #if len(masked_seq)<2:
            #    print("ok")

            # at testing
            if type(spec_label)==int:
                spec_label = torch.Tensor([spec_label]*len(masked_seq)).long()
            #    #print(spec_label)

            masked_sequence = (masked_seq, spec_label)

        """return {"masked_sequence":masked_sequence,
                "target_labels_masked":target_labels_masked,
                "target_labels":target_labels, 
                "mask": mask,
                "motif_mask": motif_mask,
                "species_labels": spec_label}
        """
        return masked_sequence, target_labels_masked, target_labels, mask, motif_mask
