#!/usr/bin/env bash

# args: task_name, max_train_samples, epochs, warmup_ratio, bsz, num_gpus, learning_rate, model_name_or_path, port

max_train_samples=$2
epochs=$3
warmup_ratio=$4
bsz=$5
num_gpus=$6
learning_rate=$7
model_name_or_path=$8
port=$9

# we log at the end of every epoch
logging_steps=$((max_train_samples / (bsz * num_gpus)))

for seed in "0"
do
    for data_seed in "0" "1" "2" "3" "4" "5" "6" "7" "8" "9"
    do
        deepspeed \
            --include localhost:0,1,2,3,4,5,6,7 \
            --master_port $port \
            $PROJECT_DIR/ft.py \
            --wandb_project_name llmft-experiments \
            --wandb_group_name vanilla-ft \
            --model_name_or_path $model_name_or_path \
            --cache_dir $HF_MODELS_CACHE \
            --task_name $1 \
            --pattern "{text1} {text2} ?" \
            --dataset_cache_dir $HF_DATASETS_CACHE \
            --max_seq_length 256 \
            --output_dir $OUTPUT_DIR \
            --overwrite_output_dir \
            --do_train \
            --max_train_samples $max_train_samples \
            --per_device_train_batch_size $bsz \
            --gradient_accumulation_steps 1 \
            --num_train_epochs $epochs \
            --warmup_ratio $warmup_ratio \
            --logging_first_step true \
            --logging_steps $logging_steps \
            --learning_rate $learning_rate \
            --weight_decay 0.0 \
            --do_eval \
            --evaluation_strategy epoch \
            --per_device_eval_batch_size 10 \
            --eval_on_hans \
            --save_strategy no \
            --fp16 \
            --seed $seed \
            --data_seed $data_seed \
            --deepspeed $PROJECT_DIR/deepspeed_configs/ds_config_zero3.json \
            --deepspeed_stage 3 \
            --report_to "none"
    done
done