from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torchvision.transforms import Lambda

import hydra

from src.datamodules.dna_datasets import CSVDataset

from src import utils
from src.datamodules.sequence_operations import compute_target_seq, encode_and_pad

log = utils.get_logger(__name__)


class MotifDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        dataset: Dataset, 
        data: dict = {
            "train_file": "data/",
            "test_file": "data/",
        },
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 512,
        num_workers: int = 4,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        transforms = None,
        test_transforms = None,
        batched_dataset: bool = False
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        # self.transforms = transforms.Compose(
        #    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        # )
        #self.transforms = encode_and_pad

        # set collate fn if batched dataset
        # https://discuss.pytorch.org/t/where-to-put-data-transforms-dataset-or-collate-fn/109868/2
        if batched_dataset:
            self.collate_fn=(lambda batch: batch[0])
        else:
            self.collate_fn = None

        self.target_transforms = hydra.utils.instantiate(test_transforms) #compute_target_seq
        self.transforms = hydra.utils.instantiate(transforms) #encode_and_pad 
        #hydra.utils.instantiate(transforms) #slow and GPU error?
        """self.transforms = Lambda(lambda sequence: encode_and_pad(
                                                                sequence,
                                                                seq_len=seq_len,
                                                                total_len=seq_len_with_pad,
                                                                mask_rate=mask_rate
                                                                ))
        self.target_transforms = Lambda(lambda sequence: compute_target_seq(
                                                                sequence[0], sequence[1],
                                                                seq_len=seq_len,
                                                                total_len=seq_len_with_pad,
                                                                ))
        """
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        #apparently it needs to be called manually https://github.com/Lightning-AI/lightning/issues/2742
        self.setup()

    @property
    def num_classes(self) -> int:
        return 5

        # def prepare_data(self):
        """Download data if needed.
        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        # MNIST(self.hparams.data_dir, train=True, download=True)
        # MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        log.info("Setting up train and test sets")
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            # dataset class defined in hydra config
            trainset = hydra.utils.instantiate(
                self.hparams.dataset,
                self.hparams.data.train_file,
                train=True, 
                transform=self.transforms,
                batch_size=self.hparams.batch_size,
                seq_position=self.hparams.data.seq_position, 
                )

            valset = hydra.utils.instantiate(
                self.hparams.dataset,
                self.hparams.data.test_file, 
                train=False, 
                transform=self.transforms,
                batch_size=self.hparams.batch_size,
                seq_position=self.hparams.data.seq_position, 
                )


            if "species_test_label" in self.hparams.data:
                testset = hydra.utils.instantiate(
                    self.hparams.dataset,
                    self.hparams.data.test_file, 
                    train=False, 
                    transform=self.target_transforms,
                    batch_size=self.hparams.batch_size,
                    seq_position=self.hparams.data.seq_position,
                    species_test_label=self.hparams.data.species_test_label,
                    )
            else:
                testset = hydra.utils.instantiate(
                    self.hparams.dataset,
                    self.hparams.data.test_file, 
                    train=False, 
                    transform=self.target_transforms,
                    batch_size=self.hparams.batch_size,
                    seq_position=self.hparams.data.seq_position, 
                    )
                
            # set batch size to 1 if batched dataset
            if self.hparams.batched_dataset:
                self.hparams.batch_size = 1

            # using test set for test and val for now.
            # it's not too bad because testing is done differently
            self.data_train = trainset
            self.data_test = testset
            self.data_val = valset

            # dataset = ConcatDataset(datasets=[trainset, testset])
            # self.data_train, self.data_val, self.data_test = random_split(
            #     dataset=dataset,
            #     lengths=self.hparams.train_val_test_split,
            #     generator=torch.Generator().manual_seed(42),
            # )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
            collate_fn=self.collate_fn
        )
