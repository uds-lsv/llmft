#!/usr/bin/env bash

OUTPUT_DIR=/root/experiment/llmft/logfiles/in_context_eval
mkdir -p $OUTPUT_DIR

# args: task_name, num_shots, model_name_or_path, gpu, port

export PROJECT_DIR=/root/experiment/llmft

# setup basic paths
export CACHE_BASE_DIR=/root/experiment/llmft/cache
export OUTPUT_DIR=/root/experiment/llmft/logfiles

# setup wandb
export WANDB_DISABLED=false
export WANDB_API_KEY=8cab6020680320fc114c896cc94035fe8ea6f51f
export WANDB_USERNAME=llama_ft_exp
export WANDB_ENTITY=llama_ft_exp
export WANDB_CACHE_DIR=$CACHE_BASE_DIR/wandb
export WANDB_CONFIG_DIR=$WANDB_CACHE_DIR

# set variables for transformers, datasets, evaluate
export TOKENIZERS_PARALLELISM=true
export HF_DATASETS_CACHE=$CACHE_BASE_DIR/hf_datasets
export HF_EVALUATE_CACHE=$CACHE_BASE_DIR/hf_evaluate
export HF_MODULES_CACHE=$CACHE_BASE_DIR/hf_modules
export HF_MODELS_CACHE=$CACHE_BASE_DIR/hf_lms
export TRANSFORMERS_CACHE=$CACHE_BASE_DIR/transformers

# set variables for torch
export TORCH_EXTENSIONS_DIR=$CACHE_BASE_DIR/torch

# create cash dirs if they don't exist yet
mkdir -p $WANDB_CACHE_DIR
mkdir -p $WANDB_CONFIG_DIR
mkdir -p $HF_DATASETS_CACHE
mkdir -p $HF_EVALUATE_CACHE
mkdir -p $HF_MODULES_CACHE
mkdir -p $HF_MODELS_CACHE
mkdir -p $TORCH_EXTENSIONS_DIR

task_name=mnli
num_shots=16
model_name_or_path=meta-llama/Llama-2-7b-hf
gpu=1

# gpt-3 format (for mnli):

# --pattern "{text1} question: {text2} Yes or No?" \
# --target_prefix " answer: " \
# --target_tokens "ĠYes,ĠNo" \
# --separate_shots_by "\n\n" \
# --group "gpt-3" \

# minimal format:

# --pattern "{text1} {text2} ?" \
# --target_prefix " " \
# --target_tokens "ĠYes,ĠNo" \
# --separate_shots_by "\n\n" \
# --group "minimal" \

# eval-harness format (for mnli):

# --pattern "{text1} \nQuestion: {text2} True or False?" \
# --target_prefix "\nAnswer: " \
# --target_tokens "ĠTrue,ĠFalse" \
# --separate_shots_by "\n\n" \
# --group "eval-harness" \

for data_seed in 0
do
    deepspeed \
        $PROJECT_DIR/eval.py \
        --model_name_or_path $model_name_or_path \
        --cache_dir $HF_MODELS_CACHE \
        --task_name $task_name \
        --pattern "{text1} question: {text2} Yes or No?" \
        --target_prefix " answer: " \
        --target_tokens "▁Yes,▁No" \
        --separate_shots_by "\n\n" \
        --group "gpt-3" \
        --dataset_cache_dir $HF_DATASETS_CACHE \
        --max_seq_length 2048 \
        --output_dir $OUTPUT_DIR \
        --do_eval  \
        --eval_task_name "hans" \
        --per_device_eval_batch_size 1 \
        --num_shots $num_shots \
        --balanced \
        --shuffle \
        --fp16 \
        --seed 0 \
        --data_seed $data_seed \
        --deepspeed $PROJECT_DIR/deepspeed_configs/ds_config_zero3.json \
        --report_to "none"
done