#!/bin/bash

dataset="mimic_cxr"
annotation="./data/longitudinal_mimic_progression_annnotation.json"
base_dir="./data/mimic_cxr/images"
delta_file="path/to/pretrained/delta_file"

version="v1_train"
savepath="./save/$dataset/$version"


CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --test \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} \
    --test_batch_size 12 \
    --freeze_vm True \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 100 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 12 \
    --devices 1 \
    --llama_model "BioMistral/BioMistral-7B" \
    --vision_model "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224" \
    --gumble_threshold 0.2 \
    --gumble_tau 0.3 \
    --seed 42 \
    2>&1 |tee -a ${savepath}/log.txt