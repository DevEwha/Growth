#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

MODEL="Enoch/llama-7b-hf"
ALPHA=0.15
PRUNE_METHOD="wanda_dlp"
SPARSITY_RATIO=0.7
SPARSITY_TYPE="unstructured"

python   run.py \
    --model $MODEL \
    --alpha $ALPHA \
    --prune_method $PRUNE_METHOD \
    --sparsity_ratio $SPARSITY_RATIO \
    --sparsity_type $SPARSITY_TYPE \