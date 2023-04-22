# icecube-2nd-place

Code for 2nd Place Icecube competition

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
