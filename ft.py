#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys

import datasets
from datasets import ClassLabel, Value
import numpy as np
import torch

import evaluate
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from options import DataTrainingArguments, ModelArguments, WandbArguments, FtArguments
from utils import create_dir, get_timestamp
from task_utils import task_to_keys, load_glue_datasets, load_hans_dataset, load_mnli_mismatched_dataset, load_paws_qqp_dataset, load_cola_ood_dataset, save_dataset
from ft_trainer import FtTrainer
from models.gptj_wrapper import GPTJWithClassifier, GPTJWithLMClassifier
from models.opt_wrapper import OPTWithClassifier, OPTWithLMClassifier
from models.llama_wrapper import LlamaWithLMClassifier
from models.gptneox_wrapper import GPTNeoXWithLMClassifier

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.23.1")

require_version("datasets>=1.8.0",
                "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments, WandbArguments, FtArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, wandb_args, ft_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, wandb_args, ft_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_glue", model_args, data_args)

    # Enable/disable wandb logging
    # os.environ["WANDB_DISABLED"] = f"{wandb_args.disable_wandb}"

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Create unique output dir
    TIMESTAMP = get_timestamp()

    if "llama/hf" in model_args.model_name_or_path:
        # these are our local llama weights
        name = model_args.model_name_or_path.split("/")
        MODEL_NAME = f"{name[-3]}-{name[-2]}-{name[-1]}"
    else:
        MODEL_NAME = model_args.model_name_or_path.replace('/', '-')
    run_name = f"{MODEL_NAME}_{data_args.task_name}-{data_args.max_train_samples}_{training_args.data_seed}_{training_args.seed}_{TIMESTAMP}"

    training_args.output_dir = create_dir(
        training_args.output_dir, run_name)

    if wandb_args.wandb_output_dir is None:
        wandb_args.wandb_output_dir = create_dir(
            training_args.output_dir, "wandb")

    if wandb_args.wandb_run_name is None:
        wandb_args.wandb_run_name = run_name

    # Load training and validation datasets
    raw_datasets, label_list, num_labels, is_regression = load_glue_datasets(
        data_args, model_args)

    additional_evaluation_datasets = {}

    if data_args.eval_on_hans:
        for heuristic in ["lexical_overlap"]:
            # for heuristic in ["lexical_overlap", "subsequence", "constituent"]:
            # Load HANS subsets as additional validation data
            for label in [0, 1]:
                subset, subset_name = load_hans_dataset(
                    data_args.dataset_cache_dir, heuristic=heuristic, subcase=None, label=label)
                additional_evaluation_datasets[subset_name] = subset

    if data_args.eval_on_mnli_mismatched:
        # Load mnli mismatched validation set
        for label in [0, 1]:
            subset, subset_name = load_mnli_mismatched_dataset(
                data_args, label=label)
            additional_evaluation_datasets[subset_name] = subset

    if data_args.eval_on_paws_qqp:
        # Load PAWS QQP validation set
        subset, subset_name = load_paws_qqp_dataset(
            path=data_args.paws_qqp_file, cache_dir=data_args.dataset_cache_dir)
        additional_evaluation_datasets[subset_name] = subset

    if data_args.eval_on_cola_ood:
        # Load CoLA ood validation set
        subset, subset_name = load_cola_ood_dataset(
            path=data_args.cola_ood_file, cache_dir=data_args.dataset_cache_dir)
        additional_evaluation_datasets[subset_name] = subset

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Add vanilla fine-tuning specific args to the model config
    config.classifier_type = ft_args.classifier_type

    # Add pattern-verbalizer fine-tuning specific args to the model config
    config.untie_embeddings = ft_args.untie_embeddings

    # Add adapter specific args to the model config
    config.use_adapters = ft_args.use_adapters
    config.adapter_type = ft_args.adapter_type
    config.adapter_dim = ft_args.adapter_dim
    config.lora_alpha = ft_args.lora_alpha

    # Add soft prompt tuning specific args to the model config
    config.use_soft_prompt = ft_args.use_soft_prompt
    config.num_soft_prompt_tokens = ft_args.num_soft_prompt_tokens

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if "gpt-j" in model_args.model_name_or_path:
        if ft_args.target_tokens is not None:
            model = GPTJWithLMClassifier.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
            )
        else:
            model = GPTJWithClassifier.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
            )

        # We need to add a padding token for gpt-j
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = tokenizer.eos_token_id

    elif "facebook/opt" in model_args.model_name_or_path:
        if ft_args.target_tokens is not None:
            model = OPTWithLMClassifier.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
            )
        else:
            model = OPTWithClassifier.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
            )

    elif "gpt-neox" in model_args.model_name_or_path or "pythia" in model_args.model_name_or_path or "RedPajama-INCITE" in model_args.model_name_or_path:
        if ft_args.target_tokens is not None:
            model = GPTNeoXWithLMClassifier.from_pretrained(
                model_args.model_name_or_path,
                from_tf=False,
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
                torch_dtype=torch.float16,
            )

            # We need to add a padding token for gptneox
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.convert_tokens_to_ids(
                tokenizer.pad_token)
            tokenizer.padding_side = "right"

        else:
            raise NotImplementedError(
                f"Unsupported model_name_or_path: {model_args.model_name_or_path}")
    
    elif "llama" in model_args.model_name_or_path:
        if ft_args.target_tokens is not None:
            model = LlamaWithLMClassifier.from_pretrained(
                model_args.model_name_or_path,
                from_tf=False,
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
                torch_dtype=torch.float16,
            )

            # We need to add a padding token for llama
            tokenizer.pad_token = tokenizer._convert_id_to_token(
                config.pad_token_id)  # let's use the <unk> token
            tokenizer.padding_side = "right"

        else:
            raise NotImplementedError(
                f"Unsupported model_name_or_path: {model_args.model_name_or_path}")

    else:
        raise NotImplementedError(
            f"Unsupported model_name_or_path: {model_args.model_name_or_path}")

    # --------------- Preprocessing the raw_datasets ---------------

    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [
            name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(
            num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {
            k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {
                i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {
            id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {
            id: label for label, id in config.label2id.items()}

    if ft_args.target_tokens is not None and not ft_args.target_tokens_logits_only:
        # we need to convert the label ids to target ids
        target_tokens = [t.strip() for t in ft_args.target_tokens.split(",")]
        target_tokens_ids = tokenizer.convert_tokens_to_ids(target_tokens)

        model.config.label2id = {
            l: target_tokens_ids[i] for i, l in enumerate(label_list)}
        model.config.id2label = {
            id: label for label, id in config.label2id.items()}

    # Compute max_seq_length
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts

        # Apply a pattern to the inputs
        pattern_examples = [
            ft_args.pattern.format(
                text1=examples[sentence1_key][idx],
                text2=examples[sentence2_key][idx] if sentence2_key is not None else None)
            for idx in range(len(examples[sentence1_key]))
        ]
        args = (pattern_examples,)
        result = tokenizer(*args, padding=padding,
                           max_length=max_seq_length, truncation=True)

        # Get mask for soft prompt tokens
        # TODO(mm): For GPT-J and GPT-NeoX we have a different tokenizer. Adjust accordingly
        if "opt" in model_args.model_name_or_path:
            # For OPT models, the first token is always the bos token </s>
            # Which happens to be also the unk token we use to mark soft prompt tokens
            # Hence, we have to be careful about which tokens to mask as part of the soft prompt
            result["soft_prompt_mask"] = [[0 if (idx != tokenizer.unk_token_id or pos == 0) else 1 for pos, idx in enumerate(indices)]
                                          for indices in result["input_ids"]]  # <unk> is the placeholder for prompt embeddings

        # Get tokens
        result["input_tokens"] = [tokenizer.convert_ids_to_tokens(
            ids) for ids in result["input_ids"]]

        # Decode input
        result["input_text"] = [tokenizer.decode(
            ids) for ids in result["input_ids"]]

        # Replace labels by target tokens indices when using lm_head
        # - special case: when using target logits only, we keep class indices instead of token indices
        if ft_args.target_tokens is not None and not ft_args.target_tokens_logits_only:
            result["label"] = [target_tokens_ids[l] for l in examples["label"]]
        else:
            result["label"] = examples["label"]

        result["label_text"] = [model.config.id2label[l] if l != -1 else "unlabeled"
                                for l in result["label"]]

        return result

    # We need to update the number of classes of the dataset when using the lm_head
    if ft_args.target_tokens is not None and not ft_args.target_tokens_logits_only:
        for split in raw_datasets:
            # raw_datasets[split].features["label"].num_classes = len(tokenizer)
            # raw_datasets[split].features["label"].names = [
            #     f"{idx}" for idx in np.arange(len(tokenizer))]

            new_features = raw_datasets[split].features.copy()
            names = [f"{idx}" for idx in np.arange(len(tokenizer))]
            new_features["label"] = ClassLabel(
                names=names, num_classes=len(tokenizer))
            raw_datasets[split] = raw_datasets[split].cast(new_features)

        for name, dataset in additional_evaluation_datasets.items():
            # dataset.features["label"].num_classes = len(tokenizer)
            # dataset.features["label"].names = [
            #     f"{idx}" for idx in np.arange(len(tokenizer))]

            new_features = dataset.features.copy()
            names = [f"{idx}" for idx in np.arange(len(tokenizer))]
            new_features["label"] = ClassLabel(
                names=names, num_classes=len(tokenizer))
            additional_evaluation_datasets[name] = dataset.cast(new_features)

    # before running the pre-processing, subsample datsets if specified

    # subsample datasets (if specified)

    # we fix the random seed that controls the sampling of the training data
    np.random.seed(training_args.data_seed)

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            # randomly select a subset of the training data
            max_train_samples = min(
                len(train_dataset), data_args.max_train_samples)
            indices = np.random.choice(
                range(len(train_dataset)), size=max_train_samples, replace=False)
            train_dataset = train_dataset.select(indices)

    if training_args.do_eval:
        # we fix the random seed that controls the sampling of the validation data
        np.random.seed(123)  # we only use this for debugging

        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name in
                                    ["mnli", "mnli-original"] else "validation"]

        # (optional) subsample eval datasets
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(
                len(eval_dataset), data_args.max_eval_samples)
            # randomly select a subset of the eval data
            indices = np.random.choice(
                range(len(eval_dataset)), size=max_eval_samples, replace=False)
            eval_dataset = eval_dataset.select(indices)

        for name, dataset in additional_evaluation_datasets.items():
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(
                    len(dataset), data_args.max_eval_samples)
                # randomly select a subset of the eval data
                indices = np.random.choice(
                    range(len(dataset)), size=max_eval_samples, replace=False)
                dataset = dataset.select(indices)
                additional_evaluation_datasets[name] = dataset

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        # we fix the random seed that controls the sampling of the validation data
        np.random.seed(123)  # we only use this for debugging

        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name in
                                       ["mnli", "mnli-original"] else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(
                range(max_predict_samples))

    # set all random seeds again (not sure if this is really needed)
    set_seed(training_args.seed)

    # tokenize and encode datasets
    with training_args.main_process_first(desc="dataset map pre-processing"):
        if training_args.do_train:
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                batch_size=1000,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on training dataset",
            )

        if training_args.do_eval:
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                batch_size=1000,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

        if training_args.do_predict:
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                batch_size=1000,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on test dataset",
            )

        for name, dataset in additional_evaluation_datasets.items():
            if "hans" in name:
                sentence1_key, sentence2_key = task_to_keys["hans"]
            elif "mnli" in name:
                sentence1_key, sentence2_key = task_to_keys["mnli"]
            elif "paws-qqp" in name:
                sentence1_key, sentence2_key = task_to_keys["paws-qqp"]
            elif "cola-ood" in name:
                sentence1_key, sentence2_key = task_to_keys["cola-ood"]

            dataset = dataset.map(
                preprocess_function,
                batched=True,
                batch_size=1000,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Running tokenizer on {name} validation dataset",
            )
            additional_evaluation_datasets[name] = dataset

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 1):
            print(
                f"Sample {index} of the training set: {train_dataset[index]}.")

    # Log training and evaluation examples to training_args.output_dir for reproducibility
    if training_args.do_train:
        save_dataset(train_dataset, path=os.path.join(
            training_args.output_dir, f"{data_args.task_name}-train.csv"))
    if training_args.do_eval:
        save_dataset(eval_dataset, path=os.path.join(
            training_args.output_dir, f"{data_args.task_name}-eval.csv"))
        for name, dataset in additional_evaluation_datasets.items():
            save_dataset(dataset, path=os.path.join(
                training_args.output_dir, f"{name}-eval.csv"))

    # --------------- End preprocessing of the raw_datasets ---------------

    # Get the metric function
    if data_args.task_name is not None:
        # use default metrics
        metric_script = f"{os.environ['PROJECT_DIR']}/metrics/glue.py"
        if data_args.task_name == "mnli-original":
            metric = datasets.load_metric(path=metric_script, config_name="mnli",
                                          cache_dir=data_args.dataset_cache_dir, keep_in_memory=False)
        else:
            metric = datasets.load_metric(path=metric_script, config_name=data_args.task_name,
                                          cache_dir=data_args.dataset_cache_dir, keep_in_memory=False)
    else:
        metric = datasets.load_metric(
            "accuracy", cache_dir=data_args.dataset_cache_dir, keep_in_memory=False)        

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(
            p.predictions, tuple) else p.predictions
        preds = np.squeeze(
            preds) if is_regression else np.argmax(preds, axis=1)

        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)

            # When using the lm_head, compute fraction of predictions that are not one of the target tokens
            if ft_args.target_tokens is not None and not ft_args.target_tokens_logits_only:
                unique_preds, counts_preds = np.unique(
                    preds, return_counts=True)
                unique_preds_counts_dict = dict(
                    zip(unique_preds, counts_preds))

                num_of_target_token_predictions = 0
                for idx in target_tokens_ids:
                    num_of_target_token_predictions += unique_preds_counts_dict.get(
                        idx, 0)
                num_other_tokens = len(
                    preds) - num_of_target_token_predictions
                result["frac_non_target_tokens"] = num_other_tokens / \
                    len(preds)

            # # Combine eval metrics
            # if len(result) > 1:
            #     result["combined_score"] = np.mean(
            #         list(result.values())).item()

            return result

        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    if training_args.do_eval:
        if len(additional_evaluation_datasets) > 0:
            # add the training task eval dataset
            additional_evaluation_datasets[data_args.task_name] = eval_dataset
            eval_datasets = additional_evaluation_datasets
        else:
            eval_datasets = eval_dataset
    else:
        eval_datasets = None
    trainer = FtTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_datasets,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        data_args=data_args,
        wandb_args=wandb_args,
        ft_args=ft_args,
        callbacks=None
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint, ignore_keys_for_eval=["past_key_values"])
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(
                train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path,
              "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_args"] = data_args.task_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
