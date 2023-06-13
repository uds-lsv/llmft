# Few-shot Fine-tuning vs. In-context Learning: A Fair Comparison and Evaluation

## Marius Mosbach, Tiago Pimentel, Shauli Ravfogel, Dietrich Klakow, Yanai Elazar

##### Saarland University, University of Cambridge, Bar-Ilan University, Allen Institute for Artificial Intelligence, University of Washington

This repository contains code for the paper [Few-shot Fine-tuning vs. In-context Learning: A Fair Comparison and Evaluation](https://arxiv.org/abs/2305.16938). 

## Abstract

Few-shot fine-tuning and in-context learning are two alternative strategies for task adaptation of pre-trained language models. Recently,
in-context learning has gained popularity over fine-tuning due to its simplicity and improved out-of-domain generalization, and because extensive evidence shows that fine-tuned models
pick up on spurious correlations. Unfortunately, previous comparisons of the two approaches were done using models of different sizes. This raises the question of whether the observed
weaker out-of-domain generalization of fine-tuned models is an inherent property of fine-tuning or a limitation of the experimental setup.
In this paper, we compare the generalization of few-shot fine-tuning and in-context learning to challenge datasets, while controlling for
the models used, the number of examples, and the number of parameters, ranging from 125M to 30B. Our results show that fine-tuned language models can in fact generalize well out-
of-domain. We find that both approaches generalize similarly; they exhibit large variation and depend on properties such as model size and the number of examples, highlighting that robust task adaptation remains a challenge.

# This repo

This repository allows to finetune (large) decoder-only language models. Currently, the following models and fine-tuning approaches are supported.

**Models**:
- [facebook/opt-125m](https://huggingface.co/facebook/opt-125m)
- [facebook/opt-350m](https://huggingface.co/facebook/opt-350m)
- [facebook/opt-1.3b](https://huggingface.co/facebook/opt-1.3b)
- [facebook/opt-2.7b](https://huggingface.co/facebook/opt-2.7b)
- [facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b)
- [facebook/opt-13b](https://huggingface.co/facebook/opt-13b)
- [facebook/opt-30b](https://huggingface.co/facebook/opt-30b)
- [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b)
- [EleutherAI/pythia-410m](https://huggingface.co/EleutherAI/pythia-410m)
- [EleutherAI/pythia-1.4b](https://huggingface.co/EleutherAI/pythia-1.4b)
- [EleutherAI/pythia-2.8b](https://huggingface.co/EleutherAI/pythia-2.8b)
- [EleutherAI/pythia-6.9b](https://huggingface.co/EleutherAI/pythia-6.9b)
- [EleutherAI/pythia-12b](https://huggingface.co/EleutherAI/pythia-12b)

**Fine-tuning approaches**:
- Vanilla fine-tuning with a randomly initialized classification head on top of the pre-trained decoder.
- Pattern-based fine-tuning (PBFT) leveraging the pre-trained language modeling head for classification.

Both of these approaches can be combined with the following paramter-efficient methods:
- BitFit (https://arxiv.org/abs/2106.10199)
- LoRA adapters (https://arxiv.org/abs/2106.09685)

## Table of Contents
1. [Setup](#setup)
2. [Memory requirements](#memory-requirements)
2. [Fine-tuning](#fine-tuning)

In order to fine-tune (very) large models (>1.3b parameters) we heavily realy on [deepspeed](https://www.deepspeed.ai/). See [memory requirements](#memory-requirements) for an estimate of the computantional resources required to fine-tune some of the models listed above. 

## Setup

1. Create docker image: 
    
        docker build -f ./docker/Dockerfile \
            --build-arg USER_UID=$UID \
            --build-arg USER_NAME=$(id -un) \
            -t llmft:22.08-py3 .

Depending on your NVIDIA CUDA and NVIDIA driver version you will have to change the `FROM nvcr.io/nvidia/pytorch:22.08-py3` line of the Docker [file](docker/Dockerfile). You can find the correct version [here](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html#framework-matrix-2022).

2. Create docker container:

        docker run -it --rm --gpus=all --pid=host --ipc=host --user <username> \
            -v <path/to/llmft>:/llmft \
            -v <path/to/datasets>:/datasets \
            -v <path/to/logfiles>:/logfiles \
            -v /<path/to/.cache>:/cache \
            llmft:22.08-py3

Make sure to replace `<username>`, `</path/to/llmft>`, etc.


# Memory requirements

We make use of deepspeed ZeRO-3 to finetune large models. Below are [estimates](https://deepspeed.readthedocs.io/en/latest/memory.html) of the memory requirements needed to train *all paramters* of the largest OPT models. We assume that you have access to a machine with at least 4 GPUs.

Also, keep in mind that these estimates consider only the model and optimizer paramters. Batch size and sequence length will also have an impact on the memory consumption during fine-tuning and inference.

## facebook/opt-6.7b

### 4 GPUs

Estimated memory needed for params, optim states and gradients for a:
- HW: Setup with 1 node, 4 GPUs per node.
- SW: Model with 6658M total params, 205M largest layer params.

|  per CPU  |  per GPU |   Options |
| --------- | --------- | --------- |
|  167.43GB |   0.77GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1 |
|  167.43GB |   0.77GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0 |
|  148.83GB |   3.87GB | offload_param=none, offload_optimizer=cpu , zero_init=1 |
|  148.83GB |   3.87GB | offload_param=none, offload_optimizer=cpu , zero_init=0 |
|    4.60GB |  28.67GB | offload_param=none, offload_optimizer=none, zero_init=1 |
|  148.83GB |  28.67GB | offload_param=none, offload_optimizer=none, zero_init=0 |

### 8 GPUs

Estimated memory needed for params, optim states and gradients for a:
- HW: Setup with 1 node, 8 GPUs per node.
- SW: Model with 6658M total params, 205M largest layer params.

|  per CPU  |  per GPU |   Options |
| --------- | --------- | --------- |
|  167.43GB |   0.77GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1 |
|  297.66GB |   0.77GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0 |
|  148.83GB |   2.32GB | offload_param=none, offload_optimizer=cpu , zero_init=1 |
|  297.66GB |   2.32GB | offload_param=none, offload_optimizer=cpu , zero_init=0 |
|    9.21GB |  14.72GB | offload_param=none, offload_optimizer=none, zero_init=1 |
|  297.66GB |  14.72GB | offload_param=none, offload_optimizer=none, zero_init=0 |


## facebook/opt-13b

### 4 GPUs

Estimated memory needed for params, optim states and gradients for a:
- HW: Setup with 1 node, 4 GPUs per node.
- SW: Model with 12853M total params, 257M largest layer params.

|  per CPU  |  per GPU |   Options |
| --------- | --------- | --------- |
|  323.21GB |   0.96GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1 |
|  323.21GB |   0.96GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0 |
|  287.30GB |   6.94GB | offload_param=none, offload_optimizer=cpu , zero_init=1 |
|  287.30GB |   6.94GB | offload_param=none, offload_optimizer=cpu , zero_init=0 |
|    5.75GB |  54.83GB | offload_param=none, offload_optimizer=none, zero_init=1 |
|  287.30GB |  54.83GB | offload_param=none, offload_optimizer=none, zero_init=0 |

### 8 GPUs

Estimated memory needed for params, optim states and gradients for a:
- HW: Setup with 1 node, 8 GPUs per node.
- SW: Model with 12853M total params, 257M largest layer params.

|  per CPU  |  per GPU |   Options |
| --------- | --------- | --------- |
|  323.21GB |   0.96GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1 |
|  574.60GB |   0.96GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0 |
|  287.30GB |   3.95GB | offload_param=none, offload_optimizer=cpu , zero_init=1 |
|  574.60GB |   3.95GB | offload_param=none, offload_optimizer=cpu , zero_init=0 |
|   11.51GB |  27.89GB | offload_param=none, offload_optimizer=none, zero_init=1 |
|  574.60GB |  27.89GB | offload_param=none, offload_optimizer=none, zero_init=0 |


## facebook/opt-30b

### 4 GPUs

Estimated memory needed for params, optim states and gradients for a:
- HW: Setup with 1 node, 4 GPUs per node.
- SW: Model with 29974M total params, 360M largest layer params.

|  per CPU  |  per GPU |   Options |
| --------- | --------- | --------- |
|  753.73GB |   1.34GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1 |
|  753.73GB |   1.34GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0 |
|  669.98GB |  15.30GB | offload_param=none, offload_optimizer=cpu , zero_init=1 |
|  669.98GB |  15.30GB | offload_param=none, offload_optimizer=cpu , zero_init=0 |
|    8.05GB | 126.96GB | offload_param=none, offload_optimizer=none, zero_init=1 |
|  669.98GB | 126.96GB | offload_param=none, offload_optimizer=none, zero_init=0 |

### 8 GPUs

Estimated memory needed for params, optim states and gradients for a:
- HW: Setup with 1 node, 8 GPUs per node.
- SW: Model with 29974M total params, 360M largest layer params.

|  per CPU  |  per GPU |   Options |
| --------- | --------- | --------- |
|  753.73GB |   1.34GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1 |
| 1339.97GB |   1.34GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0 |
|  669.98GB |   8.32GB | offload_param=none, offload_optimizer=cpu , zero_init=1 |
| 1339.97GB |   8.32GB | offload_param=none, offload_optimizer=cpu , zero_init=0 |
|   16.11GB |  64.15GB | offload_param=none, offload_optimizer=none, zero_init=1 |
| 1339.97GB |  64.15GB | offload_param=none, offload_optimizer=none, zero_init=0 |


## Fine-tuning

See [experiments/README.md](experiments/README.md) for fine-tuning instructions.
