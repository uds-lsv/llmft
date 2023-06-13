#! /bin/bash

new_devices=""
IFS=',' read -ra my_array <<< "$CUDA_VISIBLE_DEVICES"
for id in ${my_array[@]};
do
    new_devices=${new_devices}`nvidia-smi -L | grep $id | sed -E "s/^GPU ([0-9]+):.*$/\1/"`,
done
export CUDA_VISIBLE_DEVICES=${new_devices%?}
echo "now: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"