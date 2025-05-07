<h1 align="center">FRAME</h1>
<h3 align="center">Floor-aligned Representation for Avatar Motion from Egocentric Video</h3>
<h5 align="center">CVPR 2025 (Highlight)</h5>

This repository contains the official implementation of the paper "FRAME: Floor-aligned Representation for Avatar Motion from Egocentric Video".

## Table of Contents

- [Pre-requisites](#pre-requisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Getting the Dataset](#getting-the-dataset)
  - [Automatic Download](#automatic-download)
  - [Manual Download](#manual-download)
  - [Extracting](#extracting)
  - [Usage](#usage)
- [Evaluation](#evaluation)
  - [Checkpoint Download](#checkpoint-download)
- [Training](#training)
  - [Backbone Training](#backbone-training)
  - [Cross Training](#cross-training)
  - [STF Training](#stf-training)
- [Creating a new experiment](#creating-a-new-experiment)
- [CAD Models](#cad-models)
- [Citation](#citation)

## Pre-requisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) as a package manager

The code has been tested only on Debian 12.0 with NVIDIA GPUs and CUDA 11.8.  
It should work on any OS and any accelerator, as long as a bash shell is available.

## Installation

From the root folder of the repository, run the following command:

```bash
uv sync --all-extras
```

This will install all the additional dependencies required to download the dataset and extract it too.  
To activate the environment, run:

```bash
source .venv/bin/activate
```

## Dataset

This repository provides a script to download and extract the dataset used in the paper.  
Although that is the recommended approach, it is also possible to download it manually.

### Getting the Dataset

The dataset is publicly available and hosted on [Edmond](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.XARMQA).  
It can be downloaded in two different resolutions: `256x256` and `384x384`.  
Both version work with the provided code, but the `256x256` version is the one used in the paper.

#### Automatic Download

To download the dataset automatically, run the following command:

```bash
frame download dataset --output-path <path/to/output/folder> --resolution 256
```

This command will ask prompt you to accept the license agreement and then download the zip file containing the dataset.  
The script leverages `playwright` to open a browser in headless mode and download the dataset.

If you never used `playwright` before, it might ask you to install the required browsers. You can do that by running:

```bash
playwright install
```

#### Manual Download

If you prefer to download the dataset manually, you can do so by going to the [dataset page](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.XARMQA) and clicking on the `frame_v002_256.zip` entry.

#### Extracting

Given the dataset zip file, you can extract it using the provided script.

```bash
frame extract --file <path/to/dataset_256x256.zip> --output-folder <path/to/output/folder>
```

This will extract the zip file in the specified output folder and convert the `mp4` files to multiple `jpeg` images.

### Usage

We provide a helper script to manually inspect the dataset. It can be run as follows:

```bash
python scripts/loop.py --help
```

## Evaluation

In order to evaluate a model, you can run the following command:

```bash
python scripts/eval.py --data <path/to/dataset> --load <name>
```

Where `<name>` is the name of the experiment you want to evaluate, or the path to the checkpoint file.

### Checkpoint Download

In order to download the checkpoints used in the paper, you can run the following command:

```bash
frame download models
```

This will download the checkpoint files (for the backbone and the STF) in the `checkpoints` folder.  
If for any reason you want to download them manually, you can find them at the same [dataset page](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.XARMQA).

You can evaluate the model in an end2end fashion by running:

```bash
python scripts/end2end.py --data <path/to/dataset>
```

Keep in mind that in order to run `eval.py` with the STF model, you need to have the `backbone` model cached, and run the Cross Training step before.

## Training

As highlighted in the paper, the training process is divided into three steps:

1. Train the backbone model
2. Cross Caching
3. Train the STF model

### Backbone Training

```bash
python scripts/train.py --data <path/to/dataset> --experiment backbone
```

### Cross Training

Then, we do the Cross Caching (Section 4.4 of the paper).

```bash
./scripts/crosstraining.sh -d <path/to/dataset>
```

And we cache the results.

```bash
./scripts/crosscache.sh -d <path/to/dataset>
```

### STF Training

Finally, we can train the STF model.

```bash
python scripts/train.py --data <path/to/dataset> --experiment stf
```

## Creating a new experiment

This repository is based on `hydra` for configuration management.  
In order to create a new experiment, you can create a new `.yaml` file in the `configs/experiments` folder.  
That would be loaded automatically whenever you run a new training with `--experiment <name>` where `<name>` is the name of the new `.yaml` file.

A new experiment will inherit all the parameters from the `default.yaml` file, and you can override or change them from the new experiment file.

We refer to the `hydra` and `omegaconf` documentation for more details on how to use them.

## CAD Models

Instructions on how to print the CAD models can be found [here](https://github.com/abcamiletto/frame-cad).

## Acknowledgements

This project would not have been possible without some amazing open source projects. A subset of them are:

- [Pytorch Image Models](https://github.com/huggingface/pytorch-image-models)
- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Hydra](https://hydra.cc/)
- [Omegaconf](https://omegaconf.readthedocs.io/en/latest/)

## Citation

If you use this code in your research, please consider citing our paper:

```bibtex
@article{boscolo2025frame,
title = {FRAME: Floor-aligned Representation for Avatar Motion from Egocentric Video},
author = {Boscolo Camiletto, Andrea and Wang, Jian and Alvarado, Eduardo and Dabral, Rishabh and Beeler, Thabo and Habermann, Marc and Theobalt, Christian},
year = {2025},
journal={CVPR},
}
```
