import re
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import cross_entropy, nll_loss, log_softmax

import os

import torch
import numpy as np
import pandas as pd

import pyreadr

import pickle
import copy

from src.datamodules import exo_motifs

from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, precision_recall_curve
import wandb

from src import utils
from src.utils.tests import compare_sequences
from .utils import BaseRange, hot_one, outputs_to_cpu, flatten_list
from src.datamodules.motifs import sacc_cer_utr3_motif_handler as motifs3p, pombe_utr3_motif_handler as mpombe3p

log = utils.get_logger(__name__)


def positional_avg(all_motif_ce, motif_len):
    """
    cut the cross entropy list into single motifs and compute their averages
    """

    # reshape, each column, one motif 
    if len(all_motif_ce)>=motif_len:
        motif_ce_col = all_motif_ce[:len(all_motif_ce)-int(len(all_motif_ce)%motif_len)].reshape((-1,motif_len))
        avg_ce = motif_ce_col.mean(axis=1)
    else:
        avg_ce = np.array([all_motif_ce.mean()])
    

    # fix tensor []
    assert avg_ce[-1] != 0, "wrong length, computation buggy"

    return avg_ce.cpu()


class MotifMetrics():
    def __init__(
        self,
        outputs,
        wandb_log = False,
        plots_dir = os.getcwd(),
        #motif_dict = None,
        motif_dict =  motifs3p.dict, #mpombe3p.dict, # #{"TGTAAATA":1, "TGCAT":2, "ATATTC":3, "TTTTTTA":4}, # exo_motifs
        save = True  
        ) -> None:
        
        self.wandb_log = wandb_log
        self.plots_dir = plots_dir

        # concatenate all batches and sequences into one large sequence
        self.preds = torch.cat([batch["preds"].transpose(0,1).flatten() for batch in outputs])
        self.logits = torch.cat([torch.permute(batch["logits"],(2,0,1)).reshape(-1,5) for batch in outputs])
        self.targets = torch.cat([batch["targets"].transpose(0,1).flatten() for batch in outputs])
        self.motifs = torch.cat([batch["motifs"].transpose(0,1).flatten() for batch in outputs])
        #self.loss = torch.cat([batch["loss"].transpose(0,1).flatten() for batch in outputs])

        self.outputs = outputs
        
        # pass motif dict if you want to analyze specific motifs
        self.motif_dict = motif_dict

        # only keep masked ones
        self.masked_logits = self.logits[self.targets!=-100]#.cpu()
        self.masked_preds = self.preds[self.targets!=-100]#.cpu()
        self.masked_motifs = self.motifs[self.targets!=-100]#.cpu()
        self.masked_targets = self.targets[self.targets!=-100]#.cpu()
        self.ce = cross_entropy(self.masked_logits.cuda(), self.masked_targets.cuda(), reduction = "none").cuda()
        #self.ce = nll_loss(self.masked_logits.cuda(), self.masked_targets.cuda(), reduction = "none").cuda()
        
        self.residue_pos = self.annotate_residue_positions()
    
        if save:
            to_save = [outputs_to_cpu(self.outputs), self.masked_motifs.cpu(),self.masked_preds.cpu(),self.masked_targets.cpu(),self.masked_logits.cpu(),self.ce.cpu()]
            filenames = ["outputs", "masked_motifs", "masked_preds", "masked_targets", "masked_logits", "ce"]
            for tensor, file in zip(to_save,filenames):
                torch.save(tensor,file+".pt")

        self.metrics()

    def annotate_residue_positions(self):
        """
        Creates tensor of same legth as cross entropies that holds residue positions
        [0,1,2,3,0,1,2,3,4,5, ...]
        """
        positions = np.zeros(len(self.ce))
        
        # iterate through batches
        current_pos = 0
        for batch in self.outputs:
            # get sequence length
            
            batched_positions = (batch["targets"] != -100).nonzero()
            flat_positions = batched_positions[:,1]
            
            seq_len = len(flat_positions) #len(batch["preds"][0])
            #assert seq_len == len(flat_positions)-90, "length does not match: " + str(seq_len) + " " + str (len(flat_positions))
            
            # annotate with correct residue id
            positions[current_pos:current_pos+seq_len] = flat_positions.cpu()     
            
            current_pos += seq_len
        
        return positions
    
    def plot_per_residue(self, n_residues=-1):
        """
        Create per residue accuray and CE distribution plots
        """
        
        if n_residues == -1:
            n_residues = len(np.unique(self.residue_pos))
        
        # create two arrays of the num of residues
        acc = np.zeros(n_residues)
        ce = np.zeros(n_residues)
        
        # for all possible residue positions (or first n)
        for i in range(n_residues):
            # compute acc for that pos, put in array
            preds = self.masked_preds[self.residue_pos == i]
            targets = self.masked_targets[self.residue_pos == i]
            acc[i] = ((preds==targets).sum())/len(targets)
            # compute avg ce for that pos, put in array
            ce[i] = (self.ce[self.residue_pos == i]).mean()
        
        # plot the two arrays
        # create figure and axes
        fig,ax = plt.subplots()
        fig_ce,ax_ce = plt.subplots()
        
        ax.bar(list(range(n_residues)), acc, width=1.0)
        ax_ce.bar(list(range(n_residues)), ce, width=1.0)
        
        ax.set_xlabel("Residue Position")
        ax.set_ylabel("Accuracy at Residue")
        
        ax_ce.set_xlabel("Residue Position")
        ax_ce.set_ylabel("Average Cross Entropy at Residue")

        fig.savefig(self.plots_dir + "/residue_acc.png",dpi=200)
        fig_ce.savefig(self.plots_dir + "/residue_ce.png",dpi=200)

        if self.wandb_log:
            wandb.log({"test/residue_acc": fig,
                       "test/residue_ce": fig_ce,
                      })
            
        return fig_ce, ax_ce, fig, ax
            

    def plot_violin(self):
        """
        uses preds as input (logits) and targets, creates violin plot for CE based on motif
        """
        #data to be used 
        data = list(self.ce_motifs.values())
        f = open("ce_motifs.pkl","wb")
        # write the python object (dict) to pickle file
        pickle.dump(self.ce_motifs,f)
        # close file
        f.close()

        # create figure and axes
        fig,ax = plt.subplots()
        # pass axes to use
        sns.violinplot(data=data, ax=ax)

        ax.set_xticklabels(list(self.ce_motifs))

        # add random prediction CE as baseline 
        rand_train = 1.4139617494919643 # if we predict the class distribution learned from the train set
        random = 1.3862943611198906 # predict randomly (0.25%)
        class_dist = 1.3050265637015996 # predict juns class distribution

        ax.axes.axhline(y=random, color='grey', linestyle='--')
        ax.axes.axhline(y=class_dist, color='grey', linestyle='--')

        t = ax.text(
        4.7, random, "random baseline", ha="left", va="center", fontsize="x-small"
        )
        t = ax.text(
        4.7, class_dist, "class distribution baseline", ha="left", va="center", fontsize="x-small"
        )

        ax.set_ylabel("Cross Entropy averaged \n over motifs")
        fig.tight_layout()

        if self.wandb_log:
            wandb.log({"test/ce_violin_plot": fig})
        
        fig.savefig(self.plots_dir + "/violin_plot_ce.png", dpi=300)

        return fig, ax

    def plot_roc_kmer_all(self):
        """
        ROC curve, motifs vs all kmers of motif lengths
        """

        all_motif_ces = np.concatenate([self.ce_motifs[motif].cpu() for motif in self.motif_dict.keys()])
        non_motif_ces = self.ce_motifs["non_motif"].cpu()[:len(all_motif_ces)]#self.ce_kmers.values().cpu() #np.concatenate(list(self.ce_kmers.values()))
        self.ce_motifs["non_motif"] = torch.tensor(non_motif_ces).cpu()

        assert len(all_motif_ces) == len(non_motif_ces), "Should be same length" 

        # concat CEs into one array
        scores = -np.concatenate((all_motif_ces,non_motif_ces))
        labels = np.ones(len(scores))
        # set non_motif labels to 0
        labels[len(all_motif_ces):] = 0

        # compute curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        # compute AUC
        #["TGTAAATA", "TGCAT", "ATATTC", "TTTTTTA","random 5mer"]
        roc_auc = roc_auc_score(labels, scores)

        fig, ax = plt.subplots()

        ax.plot(fpr,tpr, label="motif , auc={:2.2f}".format(roc_auc)) #,label="TGTAAATA, auc={:2.2f}".format(roc_auc))
        # add diagonal
        ax.plot([0,1],[0,1], color='grey', linestyle='dashed')
        ax.legend(loc=4)

        #import pdb;pdb.set_trace()
        #display.plot()
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

        fig.savefig(self.plots_dir + "/roc_kmer_all.png",dpi=200)

        if self.wandb_log:
            wandb.log({"test/roc_kmer_all": fig})

        return fig, ax


    def plot_roc_kmer(self, motif, custom_ax=None, label=None):
        """
        ROC curve vs random kmers (same amount) of same lengths
        """
        # get motif CEs
        motif_ce = self.ce_motifs[motif].cpu()
        # get avg ces of same lengths
        non_motif_ce = self.ce_kmers[motif].cpu()
        # randomly pick n
        np.random.seed(len(motif))
        non_motif_ce = np.random.choice(non_motif_ce, size=len(motif_ce)*2)
        #self.ce_kmers[motif] = non_motif_ce

        # concat CEs into one array
        scores = -np.concatenate((motif_ce,non_motif_ce))
        labels = np.ones(len(scores))
        # set non_motif labels to 0
        labels[len(motif_ce):] = 0

        # compute curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        # compute AUC
        roc_auc = roc_auc_score(labels, scores)

        if custom_ax is None:
            fig, ax = plt.subplots()
        else:
            ax = custom_ax

        if label is None:
            label = motif +", auc={:2.2f}".format(roc_auc)
        else:
            label = label + ", auc={:2.2f}".format(roc_auc)
            
        ax.plot(fpr,tpr, label=label) #,label="TGTAAATA, auc={:2.2f}".format(roc_auc))
        # add diagonal
        ax.plot([0,1],[0,1], color='grey', linestyle='dashed')
        ax.legend(loc=4)

        if custom_ax is not None: return

        #import pdb;pdb.set_trace()
        #display.plot()
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

        fig.savefig(self.plots_dir + "/roc_kmer_" + motif + ".png",dpi=200)

        if self.wandb_log:
            wandb.log({"test/roc_kmer_"+motif: fig})

        return fig, ax
    
    def plot_roc(self, motif=None):
        """
        Get ROC of predictive power of cross entropy
        """

        # labels: motif (1) or not motif (0)
        # score: cross entropies 
        if motif!=None:
            # filter for non motif and specific motif
            filter = (self.masked_motifs==0) + (self.masked_motifs == self.motif_dict[motif])
            labels = self.masked_motifs[filter] != 0
            score = -self.ce[filter]
        else: 
            labels = self.masked_motifs!=0
            score = -self.ce


        score = score.cpu()
        labels = labels.cpu()

        fpr, tpr, thresholds = roc_curve(labels, score)

        roc_auc = roc_auc_score(labels, score)

        precision, recall, thresholds_pr = precision_recall_curve(labels, score)

        fig, ax = plt.subplots()

        if motif == None:
            motif = "all_motifs"

        ax.plot(fpr,tpr, label=motif+", auc={:2.2f}".format(roc_auc)) #,label="TGTAAATA, auc={:2.2f}".format(roc_auc))
        # add diagonal
        ax.plot([0,1],[0,1], color='grey', linestyle='dashed')
        ax.legend(loc=4)

        #import pdb;pdb.set_trace()
        #display.plot()
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")

        fig.savefig(self.plots_dir + "/roc_" + motif + ".png",dpi=200)

        if self.wandb_log:
            wandb.log({"test/roc_"+motif: fig})

        # plot PR 
        fig_pr, ax_pr = plt.subplots()

        ax_pr.plot(recall,precision, label=motif)# label=motif+", auc={:2.2f}".format(roc_auc)) #,label="TGTAAATA, auc={:2.2f}".format(roc_auc))
        # add diagonal
        # ax.plot([0,1],[0,1], color='grey', linestyle='dashed')
        ax_pr.legend(loc=4)

        #import pdb;pdb.set_trace()
        #display.plot()
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")

        fig_pr.savefig(self.plots_dir + "/pr_" + motif + ".png",dpi=200)

        if self.wandb_log:
            wandb.log({"test/pr_"+motif: fig_pr})

        return {"roc": [fig,ax], "pr":[fig_pr,ax_pr]}

    def metrics(self):
        """
        Compute needed metrics for each motif
        """

        self.ce_motifs = {}
        self.acc_motifs = {}

        # get CEs of same length as motifs (negative set)
        self.ce_kmers = {}

        # append random
        self.ce_motifs["non_motif"] = positional_avg(self.ce[self.masked_motifs==0],7)  #, 15)
        self.acc_motifs["non_motif"] = accuracy_score(
                self.masked_targets[self.masked_motifs==0].cpu(),
                self.masked_preds[self.masked_motifs==0].cpu(),
                normalize=True
                )

        if self.motif_dict is not None:
            for motif in self.motif_dict.keys():
                relevant_idx = self.masked_motifs==self.motif_dict[motif]

                if sum(relevant_idx) == 0: 
                    # nothing if no motifs found
                    continue

                # avg cross entropy per motif
                self.ce_motifs[motif] = positional_avg(
                    self.ce[relevant_idx], 
                    len(motif)
                    )

                # negative set of kmers
                self.ce_kmers[motif] = positional_avg(
                    #self.ce[~relevant_idx], 
                    self.ce[self.masked_motifs==0], 
                    len(motif)
                    )

                # accuracy per motif
                self.acc_motifs[motif] = accuracy_score(
                    self.masked_targets[relevant_idx].cpu(),
                    self.masked_preds[relevant_idx].cpu(),
                    normalize=True
                    )
        else: # for all motifs together
            self.ce_motifs["motif"] = positional_avg(self.ce[self.masked_motifs!=0],10)

            self.ce_kmers["motif"] = self.ce_motifs["non_motif"]

            self.acc_motifs["motif"] = accuracy_score(
                    self.masked_targets[self.masked_motifs!=0].cpu(),
                    self.masked_preds[self.masked_motifs!=0].cpu(),
                    normalize=True
                    )


    def summary_table(self, metrics=None):
        """
        Summarize all in dataframe for easier logging
        """
        # avg ce motifs
        df_dict = {}
        for motif in self.acc_motifs.keys():
            df_dict[motif] = [self.acc_motifs[motif], float(self.ce_motifs[motif].mean())]

        motif_df = pd.DataFrame(data=df_dict, index=["acc", "CE"])

        if metrics is not None:
            result_list = [tensor.cpu().numpy() for tensor in metrics]
            idx_list = ["acc_masked", "acc_all", "motif_acc"]
            metric_df = pd.DataFrame(data={"result": result_list}, index=idx_list)
        else:
            metric_df = None

        if self.wandb_log:
            wandb.log({"test/motif_df": motif_df})
            if metrics is not None:
                wandb.log({"test/metrics_df": metric_df})

        return motif_df, metric_df

    def plot(self, metrics=None):

        # plot roc curve for each motif
        if self.motif_dict is not None:
            for motif in self.motif_dict:
                #self.plot_roc(motif)
                #self.plot_roc_kmer(motif)
                pass
            #self.plot_roc_kmer_all()
        else:
            pass
            #self.plot_roc_kmer("motif")

        # create pandas tables
        motif_df, metric_df = self.summary_table(metrics=metrics)

        print(compare_sequences(self.masked_preds[:200], self.masked_targets[:200]))
        print(self.masked_logits[:200])

        return{
            "motif_acc": self.acc_motifs,
            "motif_ce": self.ce_motifs,
            "motif_df": motif_df,
            "metric_df": metric_df,
        }


class LoadedMotifMetrics(MotifMetrics):

    def __init__(
        self,
        load_from = "",
        motif_dict = motifs3p.dict, #{"TGTAAATA":1, "TGCAT":2, "ATATTC":3, "TTTTTTA":4}, # exo_motifs 
        compute = True,
        probas_exist = False,
        ) -> None:

        self.masked_logits = self.logits = torch.load(load_from + "masked_logits.pt")
        self.masked_preds = self.preds = torch.load(load_from + "masked_preds.pt")
        self.masked_motifs = self.motifs = torch.load(load_from + "masked_motifs.pt")
        self.masked_targets = self.targets = torch.load(load_from + "masked_targets.pt")
        self.ce = torch.load(load_from + "ce.pt")

        # compute log likelihood loss first logsoftmax
        self.sm = log_softmax(self.masked_logits, dim=1)
        self.nll = nll_loss(self.sm, self.masked_targets, reduction = "none")

        # extracting target probabilities
        #self.target_probas = torch.gather(torch.exp(self.sm), -1, self.masked_targets)
        if probas_exist:
            prbs = torch.load(load_from + "prbs.pt")
            self.target_probas = torch.gather(prbs,-1,self.masked_targets.unsqueeze(1)).squeeze() 
        else:
            self.target_probas = torch.gather(torch.exp(self.sm),-1,self.masked_targets.unsqueeze(1)).squeeze()

        self.motif_dict = motif_dict
        self.wandb_log = False
        self.plots_dir = os.getcwd()

        if compute:
            self.metrics()

    def __getitem__(self, index):
        return BaseRange(self.ce[index],
                         self.masked_targets[index],
                         self.masked_motifs[index],
                         self.masked_preds[index],
                         self.masked_logits[index])


    def map_to_df(self,df_path, seq_col="UTR3_seq"):
        if df_path.endswith(".rds"):
            self.data_df = pyreadr.read_r(df_path)[None]
        else:
            self.data_df = pd.read_csv(df_path)
        self.data_df = self.data_df[self.data_df[seq_col].notnull()].reset_index(drop=True)
        
        seq_pos = list(self.data_df[seq_col].apply(lambda x: len(x)))
        self.map_back = np.array(flatten_list([[i]*seq_pos[i] for i in range(len(seq_pos))])) 
        # create list of utrs based on df
        self.utrs=[]
        seq_starts = [0]
        for pos in seq_pos:
            self.utrs.append(self[seq_starts[-1]:seq_starts[-1] + pos])
            seq_starts.append(seq_starts[-1] + pos)


    def create_tracks(self):
        # create several files: 

        # fasta file and index for the sequences

        # write all sequences from df into one file, use NNN as delimiter between seqs
        complete_seq = ""
        for _, row in self.data_df.iterrows():
            complete_seq += row["UTR3_seq"] + "NNN"

        fasta_file = open("all.fa", "w")
        fasta_file.write(">All_concatenated\n")
        fasta_file.write(complete_seq+"\n")
        fasta_file.close()

        # bed file for predictions track
        preds_file = open("all_preds.bed", "w")
        preds_write = ""
        preds_pos = 0
        
        # cross entropy wig file for loss track
        ce_file = open("all_ce.wig", "w")
        ce_file.write("fixedStep chrom=All_concatenated start=1 step=1\n")
        ce_write = ""
        
        # wig file for motif indications
        motif_file = open("all_motifs.wig", "w")
        motif_file.write("fixedStep chrom=All_concatenated start=1 step=1\n")
        motif_write = ""

        # wig file for prediction percentage of top prediction
        top_pred_file = open("top_pred_file.wig","w")
        top_pred_file.write("fixedStep chrom=All_concatenated start=1 step=1\n")
        top_pred_write = ""

        # go through all utrs and add what we will write to the variables
        for utr in self.utrs:
            # add motif indication
            motif_write += "".join([str(float(x))+"\n" for x in utr.motifs])
            motif_write += "0.0\n0.0\n0.0\n"

            # add ce (1/CE to scale in a way that higher score is better)
            ce_write += "".join([str(round(1/float(x),4))+"\n" for x in utr.ce])
            ce_write += "0.0000\n0.0000\n0.0000\n"

            # logits to probabilities and then pick max
            probas = torch.max(torch.nn.functional.softmax(utr.logits, dim=1), dim=1)
            top_pred_write += "".join([str(round(float(x),4))+"\n" for x in probas.values])
            top_pred_write += "0.0000\n0.0000\n0.0000\n"

            # add predictions
            for pred in hot_one(utr.preds):
                preds_write += "All_concatenated\t" + str(preds_pos) +"\t" + str(preds_pos+1) + "\t" + pred + "\n"
                preds_pos += 1
            preds_pos+=3
        
        ce_file.write(ce_write)
        ce_file.close()

        preds_file.write(preds_write)
        preds_file.close()

        motif_file.write(motif_write)
        motif_file.close()

        top_pred_file.write(top_pred_write)
        top_pred_file.close()



class MetricsHandler():

    def __init__(self,
        model_paths,
        model_names,
        test_path,
        motifs = motifs3p,
        seq_col = "three_prime_region", 
        random_kmer_len = 7,
        n_random_kmers= None,
        binding_site_col = None, #"binding_range",
        existing_probas = None,
        ) -> None:
        
        # if we use several handlers, they would be based on same object
        #motifs = copy.deepcopy(motifs)
        self.motifs = motifs #copy.deepcopy(motifs)


        # load csv from test path
        if test_path.endswith(".rds"):
            self.df = pyreadr.read_r(test_path)[None]
        else:
            self.df = pd.read_csv(test_path)
        self.df = self.df[self.df[seq_col].notnull()].reset_index(drop=True)
        # this is an easter egg. if you find this, you win 100 dollars. Just kidding :P But have fun reading the code :)

        # map to start positions
        self.seq_col = seq_col
        seq_pos = list(self.df[self.seq_col].apply(lambda x: len(x)))
        self.seq_starts = [0]
        for pos in seq_pos:
            self.seq_starts.append(self.seq_starts[-1] + pos)

        # add start stop range to df
        self.df["seq_range"] = [(self.seq_starts[i],self.seq_starts[i+1]) for i in range(len(self.seq_starts)-1)]

        log.info("Loading models")
        print("Loading models")
        # allow probabilities to be computed directly
        if existing_probas is not None:
            assert len(existing_probas) == len(model_paths), "Length mismatch."
            self.models = [LoadedMotifMetrics(load_from=p, compute=False, probas_exist=prob) for p,prob in zip(model_paths, existing_probas)]
        else:
        # load metrics objects for all models
            self.models = [LoadedMotifMetrics(load_from=p, compute=False) for p in model_paths]
        self.model_names = model_names

        log.info("Adding model metrics to df")
        print("Adding models to df")
        # add to general df
        for m, name in zip(self.models,self.model_names):

            print(name)
            print(len(m.preds))

            self.df["preds_"+name] = [hot_one(np.array(m.preds[self.seq_starts[i]:self.seq_starts[i+1]])) for i in range(len(self.seq_starts)-1)]
            self.df["ce_"+name] = [np.array(m.ce[self.seq_starts[i]:self.seq_starts[i+1]]) for i in range(len(self.seq_starts)-1)]
            self.df["ll_"+name] = [np.array(-m.nll[self.seq_starts[i]:self.seq_starts[i+1]]) for i in range(len(self.seq_starts)-1)]
            self.df["target_probas_"+name] = [np.array(m.target_probas[self.seq_starts[i]:self.seq_starts[i+1]]) for i in range(len(self.seq_starts)-1)]

        log.info("Searching for motifs")
        print("Searching for motifs")
        #get complete sequence
        complete_seq = hot_one(self.models[0].targets)
        # indicating all at once - can have overlaps though
        self.m_indicator_all = np.zeros(len(self.models[0].motifs))
        # motif ranges - for motif computations later
        self.motif_ranges = []
        self.motif_ids = []

        # serach for each motif in each utr sequence
        for motif in self.motifs:
            # find all occurances
            m_indicator = np.zeros(len(self.models[0].motifs))

            for i in range(len(self.seq_starts)-1):
                start, end = self.seq_starts[i], self.seq_starts[i+1]
                seq = complete_seq[start:end]

                for match in re.finditer(motif.regex, seq):
                    # set found positions to motif id
                    m_indicator[start+match.start():start+match.end()] = motif.id
            
            # add indicator to motif
            motif.where = m_indicator
            # compute complete indicator for metrics
            self.m_indicator_all[m_indicator!=0] = m_indicator[m_indicator!=0]

            # self indictaor for each motif
            #self.df[motif.name] = motif.where
            self.df[motif.name] = [m_indicator[self.seq_starts[i]:self.seq_starts[i+1]] for i in range(len(self.seq_starts)-1)]

            # get motif ranges
            r = motif.ranges()
            assert motif.where is not None
            self.motif_ranges.append(r)
            self.motif_ids += [motif.name] * len(r)
            print(self.motifs.get_motif(name = motif.name) == motif)
            print(motif)
            print(self.motifs.get_motif(name = motif.name))

        # binding site ranges
        if binding_site_col is not None:
            r = list(self.df[~self.df[binding_site_col].isna()][binding_site_col])
            r = [[int(float(x)) for x in r_string[1:-1].split(',')] for r_string in r] # its a tuple but saved as strig
            
            #print(r)
            for x in r:
                self.m_indicator_all[x[0]:x[1]] = 100
            self.motif_ranges.append(r)
            self.motif_ids += [binding_site_col] * len(r)

        # add non-motif to dataframe
        # get number of motifs - get same amount of random non motif instances (say 7mers)
        print("Get non-motif kmer instances")
        retreived = 0
        self.non_motif_ranges = []
        np.random.seed(42)
        if n_random_kmers is None:
            n_random_kmers = len(self.motif_ids)
        while (retreived<n_random_kmers):
            non_m_indicator = np.zeros(len(self.m_indicator_all))
            kmer_len=random_kmer_len # the longer, the less variance we have here (plots look better), 
            #             but 7 or 8 porbably best comparson

            # get random UTR in df # not completely random for gpar binding - bc utrs appear several times
            if binding_site_col is not None:
                seq_r = self.df.sample().iloc[0].seq_range
                start,end = seq_r[0], seq_r[1]-kmer_len + 1
            else:
                utr_id = np.random.randint(0,len(self.df))
                start = self.seq_starts[utr_id]
                end = self.seq_starts[utr_id+1] - kmer_len + 1

            if end<=start:
                continue

            found=False
            tries = 0
            max_tries = 5
            while not found:
                # get random position in df
                tries +=1
                if tries > max_tries:
                    print("nope")
                    break

                # position should not overlap with any motif /sum of indication at that pos is zero
                rand_pos = np.random.randint(start,end)

                # retry if not found
                # fpund if no overlap with motifs, only small (2) overlap between each other
                if binding_site_col is not None:
                    if 100.0 not in self.m_indicator_all[rand_pos:rand_pos+kmer_len] and non_m_indicator[rand_pos:rand_pos+kmer_len].sum()<2:
                        non_m_indicator[rand_pos:rand_pos+kmer_len] = 1
                        self.non_motif_ranges.append((rand_pos,rand_pos+kmer_len))
                        found = True
                        retreived += 1
                elif self.m_indicator_all[rand_pos:rand_pos+kmer_len].sum() == 0 and non_m_indicator[rand_pos:rand_pos+kmer_len].sum()<2:
                    # else 
                    non_m_indicator[rand_pos:rand_pos+kmer_len] = 1
                    self.non_motif_ranges.append((rand_pos,rand_pos+kmer_len))
                    found = True
                    retreived += 1

                #else:
                    #print(self.m_indicator_all[rand_pos:rand_pos+kmer_len].sum(),self.m_indicator_all[rand_pos:rand_pos+kmer_len])


        self.motif_df = pd.DataFrame()

        # for each model get predictions inside motifs
        mranges = flatten_list(self.motif_ranges)
        self.motif_df["motif_range"] = self.non_motif_ranges + mranges 
        self.motif_df["motif"] = ["non_motif"] * len(self.non_motif_ranges) + self.motif_ids
        for model,name in zip(self.models, self.model_names):

            self.motif_df[name+"_preds"] = self.motif_df["motif_range"].apply(lambda x: model.preds[x[0]:x[1]])
            self.motif_df[name+"_ce"] = self.motif_df["motif_range"].apply(lambda x: model.ce[x[0]:x[1]])
            self.motif_df[name+"_ll"] = self.motif_df["motif_range"].apply(lambda x: -model.nll[x[0]:x[1]])
            self.motif_df[name+"_target_probas"] = self.motif_df["motif_range"].apply(lambda x: model.target_probas[x[0]:x[1]])
            # avg ce per motif
            self.motif_df[name+"_cross_entropy"] = self.motif_df[name+"_ce"].apply(lambda x: float(x.mean()))
            self.motif_df[name+"_avg_target_probas"] = self.motif_df[name+"_target_probas"].apply(lambda x: float(x.mean()))

        print("Creating large DataFrame")
        self.motif_df_l = pd.DataFrame()
        self.motif_df_l["motif_range"] = list(self.motif_df["motif_range"]) * len(self.models)
        self.motif_df_l["motif"] =  list(self.motif_df["motif"])  * len(self.models)
        self.motif_df_l["model"] = flatten_list([[name]*len(self.motif_df) for name in self.model_names])

        self.motif_df_l["ce"] = flatten_list([list(self.motif_df[name + "_ce"]) for name in self.model_names])
        self.motif_df_l["ll"] = flatten_list([list(self.motif_df[name + "_ll"]) for name in self.model_names])
        self.motif_df_l["target_probas"] = flatten_list([list(self.motif_df[name + "_target_probas"]) for name in self.model_names])

        self.motif_df_l["preds"] = flatten_list([list(self.motif_df[name + "_preds"]) for name in self.model_names])
        # avg ce per motif
        self.motif_df_l["cross_entropy"] = self.motif_df_l["ce"].apply(lambda x: float(x.mean()))
        self.motif_df_l["avg_ll"] = self.motif_df_l["ll"].apply(lambda x: float(x.mean()))
        self.motif_df_l["avg_target_probas"] = self.motif_df_l["target_probas"].apply(lambda x: float(x.mean()))


    def recompute_metrics(self):
        #print("Recomputing metrics for each model")
        print("Recomputing metrics for each model")
        for m in self.models:
            m.motifs = m.masked_motifs = self.m_indicator_all
            m.motif_dict = self.motifs.dict
            m.metrics()
