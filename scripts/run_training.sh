#!/usr/bin/bash

# run a training of stylegan for a given number of epochs
# for training on more GPUs, juste increase the nproc_per_node to the number of available cards

set -eux

outputs="${STYLEGAN_DATA_DIR}/enriching_with_stylegan/outputs/"
data="${STYLEGAN_DATA_DIR}/enriching_with_stylegan/data/"
config="${STYLEGAN_CODE_DIR}/gan/configs/Set_Exemple/"
stats="${STYLEGAN_DATA_DIR}/enriching_with_stylegan/stats/"

torchrun --nproc_per_node=$1 \
        --rdzv-endpoint=$(hostname):0 \
        --rdzv-backend=c10d \
        main_gan.py \
        --world_size=$1 \
        --data_dir=$data \
        --stat_dir=$stats \
        --id_file="small_train.csv" \
        --output_dir=$outputs \
        --epochs_num=5 \
        --var_names=['u','v','t2m'] \
        --config_dir=$config \
        --use_noise=True \
        --tanh_output=True