#!/bin/bash

if [ "$1" == "" ]; then
  echo "Pruning ratio is 0.5"
  sparsity_ratio=0.5
else
  echo "Pruning ratio is $1"
  sparsity_ratio=$1
fi

if [ "$2" == "" ]; then
  echo "unstructured"
  sparsity_type="unstructured"
elif [ "$2" == "unstructured" ] || [ "$2" == "4:8" ] || [ "$2" == "2:4" ]; then
  echo "$2"
  sparsity_type="$2"
else
  echo "$2, Invalid pruning ratio."
  exit 1
fi

# Set common variables
model="VideoLLaMA2.1-7B-AV"
home_dir="home_dir"
random_seed=42
cuda_device=0
token_selection="naive"

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

python ${home_dir}/TAMP/videollama2/evaluate.py \
    --model_path ${home_dir}/checkpoints/${model} \
    --data_path ${home_dir}/TAMP/datasets/avinstruct_avqa_music.json \
    --token_selection $token_selection \
    --data_folder ${home_dir}/TAMP/datasets/AVQA_music \
    --seed $random_seed \
    --nsamples 128 \
    --llm_sparsity_ratio $sparsity_ratio \
    --vit_sparsity_ratio 0 \
    --aud_sparsity_ratio 0 \
    --sparsity_type $sparsity_type \
    --prune_method magnitude \
    --save log \
    --save_model ${home_dir}/checkpoints/${model}_magnitude_${sparsity_ratio}_${sparsity_type} \
