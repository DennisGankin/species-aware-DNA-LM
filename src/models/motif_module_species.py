from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.utils import MaskedAccuracy
from src import utils

from src.plots.plots import MotifMetrics
import wandb

log = utils.get_logger(__name__)

class MotifLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.
    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 0.001,
        mask_rate: float = 0.2,
        weight_decay: float = 0.0005,
        test_plot: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.validation_average = []

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean") #, weight=torch.tensor([1/4*0.27161955, 1/4*0.21764797, 1/4*0.21911079, 1/4*0.29162169, 0.0])) #sum or mean?
        #self.criterion = torch.nn.NLLLoss(reduction="mean") #sum or mean?

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = MaskedAccuracy() #Accuracy(mdmc_average='global')
        self.train_acc_all = MaskedAccuracy()

        self.val_acc = MaskedAccuracy() #Accuracy(mdmc_average='global')
        self.val_acc_all = MaskedAccuracy()

        self.test_acc = MaskedAccuracy() #Accuracy(mdmc_average='global')
        self.test_acc_all = MaskedAccuracy() #Accuracy(mdmc_average='global')
        self.test_acc_motifs = MaskedAccuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor, xs: torch.Tensor):
        return self.net(x, xs)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch: Any):
        x, y_mask, y, _, _ = batch
        logits = self.forward(x[0],x[1])
        #logits = torch.zeros(logits.shape)
        #logits[:,3,:]=1 # set all to T 

        # compute cross entropy loss but only on masked positions 
        # cross entropy function ignores -100 indices
        # y = torch.argmax(y,dim=1) #class indices
        # y[mask==0] = -100 
        loss = self.criterion(logits, y_mask)

        preds = torch.argmax(logits, dim=1)
        return loss, preds, logits

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, _ = self.step(batch)
        targets = batch[2]
        targets_masked = batch[1]

        # log val metrics
        acc = self.train_acc(preds, targets_masked).detach() # compute only on masked nucleotides
        acc_all = self.train_acc_all(preds, targets).detach()
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc_all", acc_all, on_step=True, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        # detach everything except loss otherwise GPU error!!!!
        return {"loss": loss} #"preds": preds.detach().cpu(), "targets": targets_masked.detach().cpu()}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, _ = self.step(batch)
        targets = batch[2]
        targets_masked = batch[1]

        # log val metrics
        acc = self.val_acc(preds, targets_masked).detach() # compute only on masked nucleotides
        acc_all = self.train_acc_all(preds, targets).detach()
        self.log("val/loss", loss.detach(), on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc_all", acc_all, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss} #, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, logits = self.step(batch)
        targets = batch[2]
        targets_masked = batch[1]

        # get motif labels
        motif_mask = batch[-1]

        # log test metrics
        #acc = self.test_acc(preds, targets_masked) # acc only on masked nucleotides
        self.test_acc.update(preds, targets_masked)
        #acc_all = self.test_acc_all(preds, targets)
        self.test_acc_all.update(preds, targets)

        # mask out non motifs and get accuracy for motifs only
        motif_targets = targets.detach().clone()
        motif_targets[motif_mask == 0] = -100.0 
        motif_targets[targets_masked == -100] = -100.0 

        #motif_acc = self.test_acc_motifs(preds, motif_targets)
        self.test_acc_motifs.update(preds, motif_targets)

        #self.log("test/motifs", ,on_step=False,on_epoch=True)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        #self.log("test/acc_masked", acc, on_step=False, on_epoch=True)
        #self.log("test/acc_all", acc_all, on_step=False, on_epoch=True)
        #self.log("test/motif_acc", motif_acc, on_epoch=True) 
        # problem: each step is automatically aggreagated and the mean is computed.
        # this doesnt work for motif accuracy, because there are different numbers of targets for each batch
        # also this only works for other accuracies if all sequences have the same length 

        return {"loss": loss, "preds": preds, "logits": logits, "targets": targets_masked, "motifs": motif_mask}

    def test_epoch_end(self, outputs: List[Any]):
        motif_acc = self.test_acc_motifs.compute()
        acc = self.test_acc.compute()
        acc_all = self.test_acc_all.compute()

        if self.hparams.test_plot:
            log.info("Computing metrics and plots")
            metric_maker = MotifMetrics(outputs) 
            motif_metrics = metric_maker.plot(metrics=[acc,acc_all,motif_acc])
            self.log("test/motif_ce", motif_metrics["motif_ce"], on_step=False, on_epoch=True)
            self.log("test/single_motif_acc", motif_metrics["motif_acc"], on_step=False, on_epoch=True)
            log.info(motif_metrics["motif_df"])
            log.info(motif_metrics["metric_df"])

        self.log("test/acc_masked", acc, on_step=False, on_epoch=True)
        self.log("test/acc_all", acc_all, on_step=False, on_epoch=True)
        self.log("test/motif_acc", motif_acc, on_step=False, on_epoch=True)

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.train_acc_all.reset()

        self.test_acc.reset()
        self.test_acc_all.reset()
        self.test_acc_motifs.reset()

        self.val_acc.reset()
        self.val_acc_all.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )