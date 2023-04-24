# icecube-2nd-place

This repository contains the source code for the 2nd place solution in the [Kaggle IceCube Neutrino Detection Competition](https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice) developed by [DrHB](https://www.kaggle.com/drhabib) and [Iafoss](https://www.kaggle.com/iafoss). To reproduce our results, please follow the instructions provided below.

## Installation

We recommend using the official `nvidia` or `kaggle` Docker images with the appropriate CUDA version for the best compatibility

1. Clone the repository

```bash
git clone https://github.com/DrHB/icecube-2nd-place.git
```

2. Navigate to the repository folder:

```bash
cd icecube-2nd-place
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Data Download and Preparation

### Downloading the Competition Data

The official competition data can be obtained from the [Kaggle website](https://www.kaggle.com/c/icecube-neutrinos-in-deep-ice/data). Please download the data and place it in the data folder within the repository.

### Additional Datasets

1. Split `train_meta` `parquet` files for each batch, available for download [here](https://www.kaggle.com/datasets/solverworld/train-meta-parquet).

2. The [`ice_transparency.txt`](https://www.kaggle.com/datasets/anjum48/icecubetransparency) file, which contains information regarding the ice transparency of the IceCube detector.

### Preparing the Data

Create the `Nevents.pickle` file by executing the following command:

```bash
python prepare_data.py config.json PATH data
```

After completing the data preparation process, your `data` folder should have the following structure:

```bash
/data
     ├── Nevents.pickle
     ├── ice_transparency.txt
     ├── sample_submission.parquet
     ├── sensor_geometry.csv
     ├── test
     │   └── batch_661.parquet
     ├── test_meta.parquet
     ├── train
     │   ├── batch_1.parquet
     │   └── batch_2.parquet
     └── train_meta.parquet
```

## Training

```python
# B model 32
python train.py config.json \
       MODEL DeepIceModel \
       MODEL_KWARGS.dim 768 \
       MODEL_KWARGS.dim_base 192 \
       MODEL_KWARGS.depth 12 \
       MODEL_KWARGS.head_size 32
```

```python
# B model 64
python train.py config.json \
       MODEL DeepIceModel \
       MODEL_KWARGS.dim 768 \
       MODEL_KWARGS.dim_base 192 \
       MODEL_KWARGS.depth 12 \
       MODEL_KWARGS.head_size 64


```

```python
# B model 4 REL
python train.py config.json \
       MODEL DeepIceModel \
       MODEL_KWARGS.dim 768 \
       MODEL_KWARGS.dim_base 192 \
       MODEL_KWARGS.depth 12 \
       MODEL_KWARGS.head_size 32 \
       MODEL_KWARGS.n_rel 4

```

```python
# S + GNN
python train.py config.json \
       MODEL EncoderWithDirectionReconstructionV22 \
       MODEL_KWARGS.dim 384 \
       MODEL_KWARGS.dim_base 128 \
       MODEL_KWARGS.depth 8 \
       MODEL_KWARGS.head_size 32

```

```python
# B + GNN
python train.py config.json \
       MODEL EncoderWithDirectionReconstructionV23 \
       MODEL_KWARGS.dim 768 \
       MODEL_KWARGS.dim_base 128 \
       MODEL_KWARGS.depth 12 \
       MODEL_KWARGS.head_size 64

```
