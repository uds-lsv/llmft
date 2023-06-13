#!/usr/bin/env bash

export PROJECT_DIR=/llmft
source $PROJECT_DIR/scripts/misc/setup.sh

# args: task_name, max_train_samples, epochs, warmup_ratio, bsz, num_gpus, learning_rate, model_name_or_path, port
bash $PROJECT_DIR/scripts/pattern_verbalizer_ft/mnli/bitfit.sh mnli 128 100 0.25 8 4 1e-4 facebook/opt-1.3b 60000