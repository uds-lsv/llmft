# How to run experiments

## General arguments

Most arguments are set in the `llmft/scripts/*.sh` scripts. Make sure to look at them and adjust based on your hardware and setup. For example: 

- You can change the logging frequency with `--logging_steps`. 

- You can provide a different pattern via `--pattern`.

- You can provide a different learning rate via `--learning_rate`.
 
- You can freeze the embedding layer using `--freeze_emebddings`. Otherwise, embeddings will **always** be trained.

- ... 

### Deepspeed args

- When running multiple experiments on the same machine, you have to change the `--master_port`. 

- Using a particular set of GPUs is controlled via `--include localhost:`. E.g. with `--include localhost:0,1` your run will have access to GPU 0 and GPU 1 only. 

- You can specifiy a different deepspeed config file via `--deepspeed`.

- ...
