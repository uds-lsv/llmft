#!/usr/bin/env bash

OUTPUT_DIR=/logfiles/in_context_eval
mkdir -p $OUTPUT_DIR

# args: task_name, num_shots, model_name_or_path

task_name=$1
num_shots=$2
model_name_or_path=$3

# minimal format:

# --pattern "{text1} {text2} ?" \
# --target_prefix " " \
# --target_tokens "ĠNo,ĠYes" \
# --separate_shots_by "\n\n" \
# --group "minimal" \

# eval-harness format:

# --pattern "Question 1: {text1}\nQuestion 2: {text2}\nQuestion: Do both questions ask the same thing?" \
# --target_prefix " Answer: " \
# --target_tokens "ĠNo,ĠYes" \
# --separate_shots_by "\n\n" \
# --group "eval-harness" \

for data_seed in 0 1 2 3 4 5 6 7 8 9
do
    $PYTHON_BIN/deepspeed \
        --include localhost:0,1,2,3,4,5,6,7 \
        --master_port 60000 \
        $PROJECT_DIR/eval.py \
        --model_name_or_path $model_name_or_path \
        --cache_dir $HF_MODELS_CACHE \
        --task_name $task_name \
        --pattern "Question 1: {text1}\nQuestion 2: {text2}\nQuestion: Do both questions ask the same thing?" \
        --target_prefix " Answer: " \
        --target_tokens "ĠNo,ĠYes" \
        --separate_shots_by "\n\n" \
        --group "eval-harness" \
        --dataset_cache_dir $HF_DATASETS_CACHE \
        --max_seq_length 2048 \
        --output_dir $OUTPUT_DIR \
        --do_eval  \
        --eval_task_name "paws-qqp" \
        --eval_task_path $PROJECT_DIR/data/paws_qqp/dev_and_test.tsv \
        --per_device_eval_batch_size 10 \
        --num_shots $num_shots \
        --balanced \
        --shuffle \
        --fp16 \
        --seed 0 \
        --data_seed $data_seed \
        --deepspeed $PROJECT_DIR/deepspeed_configs/ds_config_zero3.json
done