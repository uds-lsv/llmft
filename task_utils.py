import logging
import pandas as pd
import numpy as np

from datasets import load_dataset, ClassLabel

logger = logging.getLogger(__name__)

task_to_keys = {
    # labels are: 0 (entailment), 1 (contradiction)
    "rte": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "mnli-original": ("premise", "hypothesis"),
    "mnli-mismatched": ("premise", "hypothesis"),
    "hans": ("premise", "hypothesis"),

    # labels are: 0 (not_duplicate), 1 (duplicate)
    "qqp": ("question1", "question2"),
    "paws-qqp": ("sentence1", "sentence2"),

    # labels are: 0 (not acceptable), 1 (acceptable)
    "cola": ("sentence", None),
    "cola-ood": ("sentence", None),
}

# See https://arxiv.org/abs/1902.01007 for a description of each of the subcases
HANS_SUBCASES = {
    "lexical_overlap": [
        # label = entailment
        'le_around_prepositional_phrase',
        'le_around_relative_clause',
        'le_conjunction',
        'le_passive',
        'le_relative_clause',

        # label = contradiction
        'ln_conjunction',
        'ln_passive',
        'ln_preposition',
        'ln_relative_clause',
        'ln_subject/object_swap',
    ],

    "subsequence": [
        # label = entailment
        'se_PP_on_obj',
        'se_adjective',
        'se_conjunction',
        'se_relative_clause_on_obj',
        'se_understood_object',

        # label = contradiction
        'sn_NP/S',
        'sn_NP/Z',
        'sn_PP_on_subject',
        'sn_past_participle',
        'sn_relative_clause_on_subject'
    ],

    "constituent": [
        # label = entailment
        'ce_adverb',
        'ce_after_since_clause',
        'ce_conjunction',
        'ce_embedded_under_since',
        'ce_embedded_under_verb',

        # label = contradiction
        'cn_adverb',
        'cn_after_if_clause',
        'cn_disjunction',
        'cn_embedded_under_if',
        'cn_embedded_under_verb',
    ]
}


def save_dataset(dataset, path):
    dataset_dict = {
        "idx": dataset["idx"],
        "label": dataset["label"],
        "input_text": dataset["input_text"],
    }
    df = pd.DataFrame(dataset_dict)

    logger.info(f"Saving dataset to: {path}")
    df.to_csv(path, columns=dataset_dict.keys())


def load_glue_datasets(data_args, model_args):
    # Get the datasets: specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    if data_args.task_name is not None:
        if data_args.task_name == "mnli":
            # convert to binary format (remove neutral class)
            raw_datasets = load_dataset(
                "glue", data_args.task_name, cache_dir=data_args.dataset_cache_dir)

            raw_datasets = raw_datasets.filter(
                lambda example: example["label"] != 1)

            # change labels of contradiction examples from 2 to 1
            def change_label(example):
                example["label"] = 1 if example["label"] == 2 else example["label"]
                return example
            raw_datasets = raw_datasets.map(change_label)

            # change features to reflect the new labels
            features = raw_datasets["train"].features.copy()
            features["label"] = ClassLabel(
                num_classes=2, names=['entailment', 'contradiction'], id=None)
            raw_datasets = raw_datasets.cast(
                features)  # overwrite old features
        
        elif data_args.task_name == "mnli-original":
            # convert to binary format (merge neutral and contradiction class)
            raw_datasets = load_dataset(
                path="glue", name="mnli", cache_dir=data_args.dataset_cache_dir)

            # change labels of contradiction examples from 2 to 1
            def change_label(example):
                example["label"] = 1 if example["label"] == 2 else example["label"]
                return example
            raw_datasets = raw_datasets.map(change_label)

            # change features to reflect the new labels
            features = raw_datasets["train"].features.copy()
            features["label"] = ClassLabel(
                num_classes=2, names=['entailment', 'contradiction'], id=None)
            raw_datasets = raw_datasets.cast(
                features)  # overwrite old features
            
        else:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                "glue",
                data_args.task_name,
                cache_dir=data_args.dataset_cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )

            if data_args.task_name == "qqp":
                # we subsample qqp already here because its really big
                # make sure we fix the seed here
                np.random.seed(123)
                for split in raw_datasets.keys():
                    raw_datasets[split] = raw_datasets[split].select(np.random.choice(
                        np.arange(len(raw_datasets[split])), size=1000, replace=False
                    ))
                    
    # Determine number of labels
    is_regression = data_args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    return raw_datasets, label_list, num_labels, is_regression


def load_mnli_mismatched_dataset(data_args, label=None, merge=False):
    subset = "mnli_mm"

    dataset = load_dataset(
        "glue", "mnli", split=f"validation_mismatched", cache_dir=data_args.dataset_cache_dir,)

    if not merge:
        # remove neutral class
        dataset = dataset.filter(
            lambda example: example["label"] != 1)

    # change labels of contradiction examples from 2 to 1
    def change_label(example):
        example["label"] = 1 if example["label"] == 2 else example["label"]
        return example
    dataset = dataset.map(change_label)

    # change features to reflect the new labels
    features = dataset.features.copy()
    features["label"] = ClassLabel(
        num_classes=2, names=['entailment', 'contradiction'], id=None)
    dataset = dataset.cast(
        features)  # overwrite old features

    if label is not None:  # filter dataset based on label
        dataset = dataset.filter(
            lambda example: example["label"] == label)
        subset = f"{subset}-{'entailment' if label == 0 else 'contradiction'}"

    return dataset, subset


def load_hans_dataset(cache_dir=None, heuristic=None, subcase=None, label=None):
    # heuristic = {lexical_overlap, subsequence, constituent}
    # subcase = see HANS_SUBCASES
    # label = {0 (entailment), 1 (contradiction)}

    subset = "hans"
    dataset = load_dataset(
        "hans", cache_dir=cache_dir, split="validation")

    # hans comes without indices, so we add them
    indices = list(range(len(dataset)))
    dataset = dataset.add_column(name="idx", column=indices)

    if heuristic is not None:  # filter dataset based on heuristic
        dataset = dataset.filter(
            lambda example: example["heuristic"] == heuristic)
        subset = f"{subset}-{heuristic}"

    if subcase is not None:  # filter dataset based on subcase
        dataset = dataset.filter(
            lambda example: example["subcase"] == subcase)
        subset = f"{subset}-{subcase}"

    if label is not None:  # filter dataset based on label
        dataset = dataset.filter(
            lambda example: example["label"] == label)
        subset = f"{subset}-{'entailment' if label == 0 else 'contradiction'}"

    return dataset, subset


def load_paws_qqp_dataset(path, label=None, cache_dir=None):
    # TODO(mm): there's probably a better way of doing this
    data_files = {"validation": path}
    dataset = load_dataset("csv", data_files=data_files,
                           sep="\t", cache_dir=cache_dir)
    dataset = dataset["validation"]

    subset = "paws-qqp"

    def _clean_data(sample):
        # the paws-qqp dataset was created as a stream of bytes. So every sentence starts with "b and ends with ".
        # we remove these
        sample["sentence1"] = sample["sentence1"][2:-1]
        sample["sentence2"] = sample["sentence2"][2:-1]
        return sample

    dataset = dataset.map(_clean_data, batched=False)
    dataset = dataset.rename_column("id", "idx")

    if label is not None:  # filter dataset based on label
        dataset = dataset.filter(
            lambda example: example["label"] == label)
        subset = f"{subset}-{'paraphrase' if label == 1 else 'not-paraphrase'}"

    return dataset, subset


def load_cola_ood_dataset(path, label=None, cache_dir=None):
    # TODO(mm): there's probably a better way of doing this
    data_files = {"validation": path}
    dataset = load_dataset("csv", data_files=data_files, sep="\t", column_names=[
                           'code', 'label', 'annotation', 'sentence'], cache_dir=cache_dir)
    dataset = dataset["validation"]

    # cola-ood comes without indices, so we add them
    indices = list(range(len(dataset)))
    dataset = dataset.add_column(name="idx", column=indices)

    subset = "cola-ood"

    if label is not None:  # filter dataset based on label
        dataset = dataset.filter(
            lambda example: example["label"] == label)
        subset = f"{subset}-{'acceptable' if label == 1 else 'unacceptable'}"

    return dataset, subset


def load_local_datasets(data_args, model_args, training_args):
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    # Loading a dataset from your local files.
    # CSV/JSON training and evaluation files are needed.
    data_files = {"train": data_args.train_file,
                  "validation": data_args.validation_file}
    # Get the test dataset: you can provide your own CSV/JSON test file (see below)
    # when you use `do_predict` without specifying a GLUE benchmark task.
    if training_args.do_predict:
        if data_args.test_file is not None:
            train_extension = data_args.train_file.split(".")[-1]
            test_extension = data_args.test_file.split(".")[-1]
            assert (
                test_extension == train_extension
            ), "`test_file` should have the same extension (csv or json) as `train_file`."
            data_files["test"] = data_args.test_file
        else:
            raise ValueError(
                "Need either a GLUE task or a test file for `do_predict`.")
    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")
    if data_args.train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        raw_datasets = load_dataset(
            "csv",
            data_files=data_files,
            cache_dir=data_args.dataset_cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        # Loading a dataset from local json files
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            cache_dir=data_args.dataset_cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Determine number of labels
    is_regression = raw_datasets["train"].features["label"].dtype in [
        "float32", "float64"]
    if is_regression:
        num_labels = 1
    else:
        # A useful fast method:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
        label_list = raw_datasets["train"].unique("label")
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)

    return raw_datasets, label_list, num_labels, is_regression
