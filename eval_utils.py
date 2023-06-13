import numpy as np
import pandas as pd

from task_utils import task_to_keys


def get_sample_ids(path):
    # we save examples as a .csv file which has an "idx" column
    df = pd.read_csv(path, sep=",", index_col=0)
    return df["idx"].values


def _select_subset_by_ids(dataset, indices):
    subset = dataset.select(indices)
    return subset


def _select_subset_by_idx(dataset, indices):
    dataset = dataset.filter(
        lambda s: s["idx"] in indices)
    return dataset


def get_balanced_subsets(dataset):
    subset_per_label = {}
    for label_idx, _ in enumerate(dataset.features["label"].names):
        subset_per_label[label_idx] = dataset.filter(
            lambda s: s["label"] == label_idx)
    return subset_per_label


def _select_random_subset(dataset, num_shots, balanced=False, seed=123):
    # fix seed
    np.random.seed(seed)

    if num_shots < 1:
        return [], []

    if balanced:
        assert num_shots % 2 == 0, "a balanced context requires at least one demonstartion per label"
        # select the same number of samples from every label
        indices = []  # we collect all indices here
        subset_per_label = get_balanced_subsets(dataset)

        for _, samples in subset_per_label.items():
            subset_indices = samples["idx"]
            # select num_shots // 2 samples
            subset_indices = np.random.choice(
                subset_indices, size=num_shots // 2, replace=False)
            indices += list(subset_indices)
        assert len(indices) == num_shots
    else:
        # just select a random subset of samples
        indices = np.random.choice(
            range(len(dataset)), size=num_shots, replace=False)

    # return _select_subset_by_ids(dataset, indices), indices
    return _select_subset_by_idx(dataset, indices), indices


def create_few_shot_context(
    dataset_name,
    dataset,
    num_shots,
    pattern,
    label_to_tokens,
    separate_shots_by=" ",
    description="",
    target_prefix="",
    from_indices=None,
    balanced=False,
    shuffle=False,
    seed=123
):
    assert pattern is not None
    assert label_to_tokens is not None

    # select samples from which the context will be constructed
    if from_indices is not None:
        demonstrations, indices = _select_subset_by_ids(dataset, from_indices)
    else:
        demonstrations, indices = _select_random_subset(
            dataset, num_shots, balanced, seed)

    if shuffle:
        if len(demonstrations) > 0:
            demonstrations = demonstrations.shuffle(seed)

    # create context
    context = "" if description == "" else f"{description}{separate_shots_by}"

    for sample in demonstrations:
        formated_sample = pattern.format(
            text1=sample[task_to_keys[dataset_name][0]],
            text2=sample[task_to_keys[dataset_name][1]
                         ] if task_to_keys[dataset_name][1] is not None else None
        )
        verbalized_label = label_to_tokens[sample["label"]]
        if verbalized_label.startswith("Ġ"):
            # we need to remove the leading whitespace from the target token in the context
            verbalized_label = verbalized_label[1:]

        elif verbalized_label.startswith("▁"):
            # we need to remove the leading whitespace from the target token in the context
            verbalized_label = verbalized_label[1:]

        context += f"{formated_sample}{target_prefix}{verbalized_label}{separate_shots_by}"

    return context, indices


def add_context_to_dataset(dataset_name, dataset, pattern, context):
    def _add_context(samples):
        result = {}
        modified_inputs = []
        key1, key2 = task_to_keys[dataset_name]

        for idx in range(len(samples[key1])):
            modified_input = f"{context}{pattern.format(text1=samples[key1][idx], text2=samples[key2][idx])}"
            modified_inputs.append(modified_input)

        result["modified_input"] = modified_inputs

        return result

    dataset = dataset.map(_add_context, batched=True, batch_size=100)

    return dataset
