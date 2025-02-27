#!/bin/bash

dataset="mimic_cxr"
annotation="./data/longitudinal_mimic_progression_annnotation.json"
base_dir="./data/mimic_cxr/images"

version="v1_train"
savepath="./save/$dataset/$version"

if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 6 \
    --val_batch_size 12 \
    --freeze_vm True \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --learning_rate 1e-4 \
    --gradient_clip_val 1 \
    --max_length 100 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 8 \
    --devices 1 \
    --max_epochs 5 \
    --limit_val_batches 0.5 \
    --val_check_interval 0.5 \
    --num_sanity_val_steps 2 \
    --llama_model "BioMistral/BioMistral-7B" \
    --vision_model "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224" \
    --gumble_threshold 0.2 \
    --gumble_tau 0.3 \
    --seed 42 \
    2>&1 |tee -a ${savepath}/log.txt
