import pandas as pd
import numpy as np
from markov_model import *

halflife_df = pd.read_csv("../../../data/Scer_half_life.csv")
all_spec = pd.read_csv("../../../data/intergenetic_regions_ensembl_53_species_train_no_saccs.csv",
                      usecols=["species","three_prime_region"])

# train scer markov
kmer_scer = KmerCount(11,pseudocount=0)
kmer_scer.compute_counts(halflife_df["UTR3_seq"])
mkv_scer = MarkovChain(kmer_scer)
mkv_scer.compile_from_counts()
mkv_scer.serialize("markov_matrix_s_cerevisiae.npy")
bimkv_scer = BiMarkov(kmer_scer)
bimkv_scer.compile_from_counts()
bimkv_scer.serialize("markov_bimatrix_s_cerevisiae.npy")

# train glabrata markov
kmer_glabrata = KmerCount(11,pseudocount=0.01)
kmer_glabrata.compute_counts(all_spec.query('species == "candida_glabrata"')["three_prime_region"])
kmer_glabrata.serialize_counts("markov_counts_c_glabrata.pkl")
mkv_glabrata = MarkovChain(kmer_glabrata)
mkv_glabrata.compile_from_counts()
mkv_glabrata.serialize("markov_matrix_c_glabrata.npy")
bimkv_glabrata = BiMarkov(kmer_glabrata)
bimkv_glabrata.compile_from_counts()
bimkv_glabrata.serialize("markov_bimatrix_c_glabrata.npy")

# train all funghi markov
kmer_all = KmerCount(11,pseudocount=0.01)
kmer_all.compute_counts(all_spec["three_prime_region"])
kmer_all.serialize_counts("markov_counts_all.pkl")
mkv_all = MarkovChain(kmer_glabrata)
mkv_all.compile_from_counts()
mkv_all.serialize("markov_matrix_all.npy")
bimkv_all = BiMarkov(kmer_all)
bimkv_all.compile_from_counts()
bimkv_all.serialize("markov_bimatrix_all.npy")