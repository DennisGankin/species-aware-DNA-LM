# Species-aware DNA language modeling


This repository accompanies the paper "Species-aware DNA language modeling". It contains the source code for the Masked Language Model from the paper, pre-trained models and baselines, and the test data. Feel free to contact us if you have any questions, or create a github issue. If you use this in your research please cite:


Gankin, Karollus et al. (2023) "Species-aware DNA language modeling" 

## Requirements

Create a conda environment from the ``environment.yml`` file:
```
conda spade create -f environment.yml
```

To run anything in the repo, activate the environment:
```
conda activate spade
```

## Data

As the data is too large for GitHub, please download it from https://zenodo.org/record/7569953 into the ``data/`` directory. 
The download provides model checkpoints, baselines models, evaluation results and datasets used for training, testing and downstream tasks.

## Model

Model checkpoints are saved in ``data/models/``. 

### Reproducing paper results
To reproduce results from the paper, we provide predefined experiment runs.
Results will be stored in automatically created output files, handled by hydra.
The experiments simplify configuration by pre-defining the most important parameters.
Run an experiment with this command: ``python test.py experiment=experiment_name --multirun``. 

#### Evaluating baselines

To evaluate the specific-agnostic SSM and the S.cer specific di-nucleotide model, run:
```
python test.py experiment=test_baselines_in_scer --multirun
```

To evaluate k-mer models fitted on the train set, run:
```
python test.py experiment=test_markov_baselines --multirun
```
 
#### Evaluating the species-aware SSM model

In S.cer with two different species labels:
```
python test.py experiment=test_species_in_scer --multirun
```

In N.crassa with two different species labels:
```
python test.py experiment=test_species_in_ncrassa --multirun
```

#### Testing on gPAR CLIP data

Species-agnostic SSM:
```
python test.py experiment=gpar_zero_shot_baseline
```

Species-aware SSM:
```
python test.py experiment=gpar_zero_shot
```

K-mer models:
```
python test.py experiment=gpar_zero_shot_markov
```

#### Retreive embeddings

To retreive embeddings from the species-aware SSM, set the configured model from ``model=species_dss`` to ``model=species_dss_weights``. After testing, the extracted embeddings will be saved in the directory created by hydra for the run. 

### Testing

For inference, run the following command:
```
python test.py 
```
The results will be saved in the automatically created ``outputs/`` directory. The configuration for testing is defined in the file ``configs/test.yaml``

You can choose different models and checkpoints for inference.
For the species-agnostic SSM, set ``model=motif_s4_resnet`` and ``ckpt_path=${ssm}``.
For the species-aware SSM, set ``model=species_dss`` and ``ckpt_path=${spec_aware_ssm}``.

### Training 

For training, run the follwing command:
```
python train.py
```
The model checkpoint will automatically be saved locally. The configuration for training is defined in the file ``configs/train.yaml`

### GPU 

Running the model on a GPU drastically improves performance. 
If you want to run it only on CPU, make sure to set the ``trainer.gpus=0`` flag.

### Configuration

The ``configs`` directory contains files to configure the pipeline using hydra.
There, we define the used model, datamodule, trainer, logger and all their specific parameters.
So, changes to the parameters can be easily done there. E.g. If you want to train with a different learning rate, run: ``python train.py trainer.learning_rate=0.042``. You can also define your own model parameter set in a file and pass that in the command line: ``python train.py model=my_model``. Alternatively, you can asjust the exsisting configuartion files.

The ``model`` parameter set allows to control the architecture with parameters such as number of layers, dropout, dimensionality of hidden layers, and others. The ``datamodule`` configuration handles the input datasets, and data transformation parameters, such as sequence length.

The configuration logic and file structure is loosely adapted from this [tempate](https://github.com/ashleve/lightning-hydra-template).

## Reproducing plots

The plotting functions can be found in the ``notebooks/`` folder.
The notebooks read the results saved in ``data/results/`` and compute the plots and comparisons used for the paper figures.

