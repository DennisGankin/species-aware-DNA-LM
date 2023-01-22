from torchmetrics import Metric
import torch

class MaskedAccuracy(Metric):
    """
    Computes the Accuracy for only the masked nucleotides.
    All target values holding the ignore index will be ignored during the accuracy computation
    """
        
    # Set to True if the metric is differentiable else set to False
    is_differentiable: bool = False

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: bool = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = True

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.ignore_index = -100.0

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        #preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += torch.sum(target != self.ignore_index) # ignore masked ones

    def compute(self):
        """
        Divide correct predictions by all predictions (ignoring the masked ones)
        """
        if self.total != 0:
            return self.correct.float() / self.total
        return 0 # if we gotta divide by 0
