# setup basic paths
export CACHE_BASE_DIR=/home/mmosbach/cache
export OUTPUT_DIR=/home/mmosbach/logfiles

# setup wandb
export WANDB_DISABLED=false
export WANDB_API_KEY=
export WANDB_USERNAME=
export WANDB_ENTITY=
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

# set path to python
# export PYTHON_BIN="/home/mmosbach/miniconda3/envs/llmft/bin"

# rename GPUs
source $PROJECT_DIR/scripts/misc/rename_gpu_ids.sh

cd $PROJECT_DIR
echo $HOSTNAME
nvidia-smi