# Deepspeed ZeRO performance tuning

Disclaimer: Most of the suggestions below are taken from: https://huggingface.co/docs/transformers/main_classes/deepspeed#zero3-config

See also [here](https://www.deepspeed.ai/docs/config-json/) for further information in the individual arguments.

## OOM issues

We had some OOM issue when fine-tuning the largest OPT model (30B). We fixed them be reducing the `reduce_bucket_size` argument. 