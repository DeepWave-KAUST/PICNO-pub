# PICNO-pub
![LOGO](https://github.com/DeepWave-KAUST/VMB_with_diffusion-FNO/blob/main/network2_01.png)
Reproducible material for **An effective physics-informed neural operator framework for predicting wavefields - Xiao Ma, Tariq Alkhalifah**

# Project structure

This repository is organized as follows:

- :open_file_folder: **package**: python library containing routines for ....;
- :open_file_folder: **asset**: folder containing logo;
- :open_file_folder: **data**: folder containing data (or instructions on how to retrieve the data
- :open_file_folder: **notebooks**: set of jupyter notebooks reproducing the experiments in the paper (see below for more details);
- :open_file_folder: **scripts**: set of python scripts used to run multiple experiments ...

## Notebooks

The following notebooks are provided:

- :orange_book: `openfwi_xHz_cno_no_pde.ipynb`: notebook performing results with no pde loss;
- :orange_book: `openfwi_xHz_cno.ipynb`: notebook performing results with pde loss

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

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA A100 GPU. Different environment
configurations may be required for different combinations of workstation and GPU.
