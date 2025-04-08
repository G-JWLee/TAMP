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
model="llama3-llava-next-8b"
home_dir="home_dir"
calibration_source="sharegpt4v"
random_seed=42
cuda_device=0
token_selection="naive"

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

python llava/evaluate.py \
    --model_path ${home_dir}/checkpoints/${model} \
    --conv_mode llava_llama_3 \
    --data_path ${home_dir}/TAMP/playground/LLaVA-NeXT-Data/llava_next_raw_format/llava_next_raw_format_processed.json \
    --task_split_path ${home_dir}/TAMP/playground/LLaVA-NeXT-Data/llava_next_raw_format/task_split.json \
    --task_name $calibration_source \
    --token_selection $token_selection \
    --image_folder ${home_dir}/TAMP/playground/LLaVA-NeXT-Data/llava_next_raw_format \
    --seed $random_seed \
    --nsamples 128 \
    --llm_sparsity_ratio $sparsity_ratio \
    --vit_sparsity_ratio 0 \
    --sparsity_type $sparsity_type \
    --prune_method wanda \
    --sparsity_ratio_granularity block \
    --score_method outlier_sum \
    --save log \
    --save_model ${home_dir}/checkpoints/${model}_owl_${sparsity_ratio}_${sparsity_type} \
