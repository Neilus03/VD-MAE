#!/bin/bash

source /home/ndelafuente/miniconda3/etc/profile.d/conda.sh
conda activate vdmae

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5python -m
WORLD_SIZE=6 python -m
RANK=0 python -m

torchrun --nproc_per_node=6 pretraining.py