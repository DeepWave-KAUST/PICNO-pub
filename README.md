# PICNO-pub

![LOGO](https://github.com/DeepWave-KAUST/PICNO-pub/blob/main/Network.png)
Reproducible material for **An effective physics-informed neural operator framework for predicting wavefields - Xiao Ma, Tariq Alkhalifah**

# Project structure

This repository is organized as follows:

- :open_file_folder: **asset**: folder containing logo;
- :open_file_folder: **data**: Instructions on how to retrieve the data
- :open_file_folder: **notebooks**: set of jupyter notebooks reproducing the experiments in the paper (see below for more details);
- :open_file_folder: **neuralseismic_xiao**: set of python scripts used to run multiple experiments ...

## Notebooks and python file

The following notebooks or file are provided:

- :orange_book: `openfwi_xHz_cno_no_pde.ipynb`: notebook performing results with no pde loss;
- :orange_book: `openfwi_xHz_cno.ipynb`: notebook performing results with pde loss
- :snake: train_cno.py: training script for the CNO-based model (handles data loading, model initialization, loss definition, and the full training/validation loop).

## Getting started :space_invader: :robot:

To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:

```
./install_env.sh
```

It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go.

Remember to always activate the environment by typing:

```
conda activate my_env
```

**Disclaimer**: All experiments were conducted on a SLURM-managed GPU cluster equipped with Intel® Xeon® CPUs @ 2.10 GHz and a single NVIDIA A100 GPU per job allocation. Different environment configurations may be required for other workstation, cluster, or GPU architectures.
