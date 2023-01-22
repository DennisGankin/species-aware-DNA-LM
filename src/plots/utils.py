import pickle
import torch
import numpy as np
import pandas as pd

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import logomaker

def load_pickle(dir_path, filenames):
    """Loads objeckts from pickle files and returns the objects.

    Args:
        dir_path (str): Path to directory where pickle files are located
        filenames (list[str] or str): list of all filenames to be loaded, or just one filename as str, filenames must not include the .pkl part

    Returns:
        _type_: list of objects loaded from pickle files or single object if one file passed as string to filenames
    """    

    # add / to path, if not there
    if dir_path!="" and dir_path[-1]!="/":
        dir_path += "/"

    if type(filenames) == list:
        loaded_objects = []
        for file in filenames:
            with open(dir_path + file + ".pkl", "rb") as file:
                object = pickle.load(file)
                loaded_objects.append(object)
        return loaded_objects

    else: # type(filenames) == str
        with open(dir_path + filenames + ".pkl", "rb") as file:
            object = pickle.load(file)
        return object


def save_pickle(filenames, objects_to_save, dir_path=""):
    """Saves given objects to pickle files with specified names.

    Args:
        filenames (list[str] or str): Filenames to save objects as or sinngle filename as string.
        objects_to_save (_type_): Either list of objects to save, should have same length as filenames list, or single object.
        dir_path (str, optional): Directory path for pickle files to be saved in. Defaults to "".
    """

    # add / to path, if not there
    if dir_path!="" and dir_path[-1]!="/":
        dir_path += "/"

    if type(filenames) == list:
        assert type(objects_to_save) == list, "Objects you want to save where not passed as a list."
        assert len(filenames) == len(objects_to_save), "The amount of objects to save ("+ str(len(objects_to_save)) +") did not match the amount of given filenames ("+ str(len(filenames)) +")"

        for file, object in zip(filenames,objects_to_save): 
            f = open(dir_path + file + ".pkl","wb")
            pickle.dump(object,f)
            f.close()

    else: # type(filenames) == str
        f = open(dir_path + file + ".pkl","wb")
        pickle.dump(objects_to_save,f)
        f.close()

def outputs_to_cpu(outs):
    """Goes through output list and maps all tensors in each batch to cpu.

    Args:
        outs (list[dict]): Output list containing batches/ dicts of tensors

    Returns:
        list[dict]: Output list of batches with all tensors on cpu
    """    

    for batch in outs:
        for k in batch.keys():
            batch[k] = batch[k].cpu()

    return outs


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    BLACK = '\033[30m'
    WHITE = '\033[38;2;255;255;255m' #\033[37m'

## background
class bgcolor:
    BLACK = '\033[40m'
    RED = '\033[41m'
    GREEN = '\033[42m'
    WHITE = '\033[47m'
    BLUE = '\033[44m'
    WHITE = '\033[48;2;255;255;255m' #'\033[47m'
    
mapping = "ACGTN"
    
def compare_sequences_with_motif(pred, target, motif):
    pred_string = "pred: "
    target_string = "true: "
    # indicator if we are at true or false
    for p_base, t_base, is_motif in zip(pred, target, motif):

        if p_base == -100 or t_base == -100:
            continue

        # to nucleotide
        p_nuc = mapping[p_base]
        t_nuc = mapping[t_base]
        
        # compare
        if p_nuc == t_nuc:
            # not yet set to blue
            pred_string += color.WHITE + bgcolor.GREEN
        else:
            pred_string += color.WHITE + bgcolor.RED

        # check for motif
        if is_motif:
            # make bold
            target_string += color.BOLD + color.WHITE + bgcolor.BLUE

        if not is_motif :
            # remove bold again 
            target_string += color.END #color.BLACK + bgcolor.WHITE


        # add nucleotides
        pred_string += p_nuc
        target_string += t_nuc
        
    pred_string += color.END
    target_string += color.END
    return pred_string+ "\n"+ target_string


class BaseRange():
    def __init__(self, ce, targets, motifs, preds, logits):
        self.ce = ce 
        self.targets= targets
        self.preds = preds
        self.motifs = motifs
        self.logits = logits
        self.motif_colors = ["#FFD166","#06D6A0","#EF476F","#A100FE"]
        self.motif_seqs= ["TGTAAATA", "TGCAT", "ATATTC", "TTTTTTA"]
        
        self.motif_count = torch.tensor([np.round(sum(self.motifs==i)/len(self.motif_seqs[i-1])) for i in range(1,len(self.motif_seqs)+1)])
    
    def __len__(self):
        return len(self.ce)
    
    def compare_prediction(self):
        return compare_sequences_with_motif(self.preds, self.targets, self.motifs)
    
    def logo(self, start=0, stop=None, figsize=(20,3)):
        if stop is None:
            stop = len(self)
        logit_logo(self.logits[start:stop], figsize = (20,3))
        
    def plot_ce(self, start=0, stop=None):
        if stop is None:
            stop = len(self)
                
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(self.ce[start:stop])

        for i in (self.motifs[start:stop] != 0).nonzero():
            # highlight a time range
            ax.axvspan(int(i), int(i)+1, color=self.motif_colors[int(self.motifs[start+i])-1], alpha=0.3)
            
        
        # create legend
        handles = []
        for col,mot in zip(self.motif_colors, self.motif_seqs):
            patch = mpatches.Patch(color=col, label=mot, alpha=0.3)
            handles.append(patch)
            
        ax.legend(handles=handles)
        ax.set_xlim(0,stop-start)
        
        ax.set_ylabel("Avrerage Cross Entropy")
        ax.set_xlabel("Relative nucleotide position")
        
        plt.show()


def logit_logo(logits, softm=False, figsize=(15,3)):
    
    if softm:
        logits = torch.nn.functional.softmax(logits, dim=1)
    
    logits = logits[:,:4]

    logits_plot = pd.DataFrame(logits)   # channel_weights.permute(1,0).numpy())
    logits_plot.columns = ["A","C","G","T"]
    logomaker.Logo(logits_plot, figsize = figsize)
    
def hot_one(seq):
    return "".join(["ACGTN"[int(i)] for i in seq])

def flatten_list(l):
    return [item for sublist in l for item in sublist]
