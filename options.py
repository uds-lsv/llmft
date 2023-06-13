from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Optional

from task_utils import task_to_keys, HANS_SUBCASES


# We restrict fine-tuning to these models for now
SUPPORTED_MODELS = [
    # OPT models
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    "facebook/opt-66b",

    # Path to LLaMA models
    "/home/mmosbach/cache/llama/hf/7B",
    "/home/mmosbach/cache/llama/hf/13B",

    # GPT-NeoX
    "EleutherAI/gpt-neox-20b",

    # Pythia models
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
    "EleutherAI/pythia-12b-deduped",
]


@dataclass
class WandbArguments:
    disable_wandb: bool = field(
        default=False, metadata={"help": "Whether to disable wandb logging."}
    )
    # We provide these via environment variables so we comment this
    # api_key: Optional[str] = field(
    #     default=None, metadata={"help": "Your wandb API key."}
    # )
    # user_name: Optional[str] = field(
    #     default=None, metadata={"help": "Your wanbd user name."}
    # )
    # entity: Optional[str] = field(
    #     default=None, metadata={"help": "wandb enitiy. this can be your user name or a team name."}
    # )
    wandb_project_name: Optional[str] = field(
        default="llmft-experiments", metadata={"help": "The name of the wandb project."}
    )
    wandb_run_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the current run."}
    )
    wandb_group_name: Optional[str] = field(
        default=None, metadata={"help": "The group name for the current run."}
    )
    wandb_output_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to store the wandb logfiles."}
    )

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = dict((field.name, getattr(self, field.name))
                 for field in fields(self) if field.init)

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " +
                  ", ".join(task_to_keys.keys()),
                  "choices": task_to_keys.keys()
                  },
    )
    eval_task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on evaluate. Use this for in-context eval: " +
                  ", ".join(task_to_keys.keys()),
                  "choices": task_to_keys.keys()
                  },
    )
    eval_task_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the evaluation data. This has to be specified for paws-qqp and cola-ood."},
    )

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to save the cached dataset."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={
                                     "help": "A csv or a json file containing the test data."})

    # Arguments used to evaluate on additional datasets

    eval_on_hans: bool = field(
        default=False, metadata={"help": "Whether to evaluate on the HANS dataset during fine-tuning."}
    )

    # hans_heuristics: Optional[str] = field(
    #     default=None, metadata={
    #         "help": 'Comma-separated list of HANS heuristics to consider. If specified, we only consider HANS examples that are label with one of these heuristic. Choose from "lexical_overlap", "subsequence", "constituent"',
    #     }
    # )

    # hans_subcase: Optional[str] = field(
    #     default=None, metadata={"help": "If specified, we only consider HANS examples that are label with this particular subcase."}
    # )

    eval_on_mnli_mismatched: bool = field(
        default=False, metadata={"help": "Whether to evaluate on the MNLI mismatched dataset during fine-tuning."}
    )

    eval_on_paws_qqp: bool = field(
        default=False, metadata={"help": "Whether to evaluate on the PAWS-QQP dataset during fine-tuning."}
    )

    paws_qqp_file: Optional[str] = field(default="/llmft/data/paws_qqp/dev_and_test.tsv", metadata={
        "help": "A .tsv file containing the PAWS-QQP validation and test data."})

    eval_on_cola_ood: bool = field(
        default=False, metadata={"help": "Whether to evaluate on the CoLA-OOD dataset during fine-tuning."}
    )

    cola_ood_file: Optional[str] = field(default="/llmft/data/cola_ood/dev.tsv", metadata={
        "help": "A .tsv file containing the CoLA-OOD validation and test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError(
                    "Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError(
                "Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in [
                "csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."

        if self.eval_on_hans:
            assert self.task_name in [
                "rte", "mnli", "mnli-original"], "evaluation on HANS requires training on rte or mnli."

            # if self.hans_subcase is not None:
            #     assert self.hans_heuristic is not None, "specifying a subcase requires specifying a heuristic"
            # valid_subcases = HANS_SUBCASES[self.hans_heuristic]
            # assert self.hans_heuristic in valid_subcases, f"invalid subcase for the specified heuristic. valid subcases are: {valid_subcases}"

        if self.eval_on_mnli_mismatched:
            assert self.task_name in [
                "rte", "mnli", "mnli-original"], "evaluation on mnli-mismatched requires training on rte or mnli."

        if self.eval_on_paws_qqp:
            assert self.paws_qqp_file is not None, "evaluating on paws-qqp requires providing a path to the evaluation data."

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = dict((field.name, getattr(self, field.name))
                 for field in fields(self) if field.init)

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model",
            "choices": SUPPORTED_MODELS
        }
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = dict((field.name, getattr(self, field.name))
                 for field in fields(self) if field.init)

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class FtArguments:
    # General fine-tuning args
    freeze_embeddings: bool = field(
        default=False,
        metadata={
            "help": (
                "Do not update embedding parameters."
            )
        },
    )

    classifier_type: Optional[str] = field(
        default="fully-connected",
        metadata={
            "help": (
                "Which classification head to use for vanilla fine-tuning. Our default is fully-connected."
            ),
            "choices": ["linear", "fully-connected"],
        },
    )

    untie_embeddings: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to untie input and output embedding matrices."
            )
        },
    )

    log_l2_dist_per_weight: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to log the l2 distance between pre-trained and fine-tuned weights for all trainable weights."
            )
        },
    )

    # Arguments for pattern-verbalizer fine-tuning
    pattern: Optional[str] = field(
        default="{text1} {text2} ?", metadata={"help": "The input pattern. We will apply this pattern to every sample of the training and validation datasets."}
    )

    target_tokens: Optional[str] = field(
        default=None, metadata={"help": "Comma separated list of target tokens when using the lm_head for prediction, e.g. ĠYes,ĠNo"}
    )

    target_tokens_logits_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Consider only the logits of the target tokens when selecting the arg max."
            )
        },
    )

    head_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Fine-tune only the lm/classificaiton head."
            )
        },
    )

    # Arguments for bitfit fine-tuning
    bitfit: bool = field(
        default=False,
        metadata={
            "help": (
                "Fine-tune only biases."
            )
        },
    )

    # Arguments for adapter fine-tuning
    use_adapters: bool = field(
        default=False,
        metadata={
            "help": (
                "Fine-tune only adapters."
            )
        },
    )

    adapter_type: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Which adapter architecture to use."
            ),
            "choices": [None, "lora", "ia3", "parallel-attn", "parallel-fc", "parallel",
                        # "sequential-attn", "sequential-fc", "sequential"
                        ],
        },
    )

    # adapter_location: Optional[str] = field(
    #     default=None,
    #     metadata={
    #         "help": (
    #             "Where to place the adapter."
    #         ),
    #         "choices": [None, "attention", "fully-connected", "both"],
    #     },
    # )

    adapter_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Hidden dimension of the adapter layers."
            )
        },
    )

    lora_alpha: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "When using LoRA, the result of applying the adapter to the input is scaled by lora_alpha."
            )
        },
    )

    use_soft_prompt: bool = field(
        default=False,
        metadata={
            "help": (
                "Fine-tune only soft prompt parameters."
            )
        },
    )

    num_soft_prompt_tokens: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of trainable embeddings to include in the soft prompt."
            )
        },
    )

    # deepspeed stage
    deepspeed_stage: Optional[int] = field(
        default=2,
        metadata={
            "help": (
                "When using deepspeed specify the stage to support weight logging."
            ),
            "choices": [None, 2, 3],
        },
    )

    def __post_init__(self):
        # Sanity checks for fine-tuning arguments
        assert not (
            self.bitfit and self.use_adapters), "using bitfit together with adapters is not supported."

        if self.use_adapters:
            assert self.adapter_type is not None, "when using adapters you need to specify an --adapter_type."
            if self.adapter_type == "lora":
                assert self.lora_alpha is not None, "when using LoRA, --lora_alpha needs to be specified."
            # assert self.adapter_location is not None, "when using adapters you need to specify an --adapter_location."
            assert self.adapter_dim is not None or self.adapter_type == "ia3", "when using adapters you need to specify an --adapter_dim."

        if self.use_soft_prompt:
            assert ("</s>" in self.pattern or "<unk>" in self.pattern), "when using soft prompts, make sure to specify placeholder tokens in the pattern."

            if "</s>" in self.pattern:
                assert self.pattern.count(
                    "</s>") == self.num_soft_prompt_tokens, "--num_soft_prompt_tokens and number of placeholder tokens in the pattern have to agree."
            if "<unk>" in self.pattern:
                assert self.pattern.count(
                    "<unk>") == self.num_soft_prompt_tokens, "--num_soft_prompt_tokens and number of placeholder tokens in the pattern have to agree."

            assert self.num_soft_prompt_tokens > 0, "when using soft prompts, make sure to set --num_soft_prompt_tokens > 0."

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = dict((field.name, getattr(self, field.name))
                 for field in fields(self) if field.init)

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class InContextLearningArguments:
    task_description: Optional[str] = field(
        default="", metadata={"help": "A description added to the beginning of the context"}
    )

    pattern: Optional[str] = field(
        default="{text1} {text2} ?", metadata={"help": "The input pattern. We will apply this pattern to every sample of the training and validation datasets."}
    )

    target_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to be added before the target token."}
    )

    target_tokens: Optional[str] = field(
        default=None, metadata={"help": "Comma separated list of target tokens when using the lm_head for prediction, e.g. ĠYes,ĠNo"}
    )

    target_tokens_logits_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Consider only the logits of the target tokens when selecting the arg max."
            )
        },
    )

    num_shots: int = field(
        default=0,
        metadata={
            "help": (
                "Total number of demonstrations in the context."
            )
        },
    )

    separate_shots_by: Optional[str] = field(
        default=" ", metadata={"help": "How to separate demonstartions in the prompt. Default is empty space."}
    )

    sample_indices_file: Optional[str] = field(
        default=None, metadata={"help": "Path to a file that contains indices for demonstrations."}
    )

    balanced: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to choose an equal number of demonstrations from all available classes"
            )
        },
    )

    shuffle: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to shuffle the demonstrations in the context"
            )
        },
    )

    group: Optional[str] = field(
        default="None", metadata={"help": "A unique group name for a set of evaluations."}
    )

    num_data_seeds: int = field(
        default=1,
        metadata={
            "help": (
                "How many data seeds to use."
            )
        },
    )
