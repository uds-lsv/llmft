# Datasets

We currently support training on the following datasets:

- RTE
- MNLI
- QQP
- CoLA

Evaluation can be performed on:

- The corresponding validation sets
- HANS
- PAWS-QQP
- CoLA-OOD

Datasets are loaded automatically via [huggingface datasets](https://huggingface.co/datasets/glue). The PAWS-QQP and CoLA-OOD datasets are provided inside this folder.

## Required disk space

Storing the entire GLUE dataset (which contains RTE, MNLI, QQP, and CoLA) requires around 1.24GB.

