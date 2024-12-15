#!/usr/bin/bash

# run an inversion on a given set with a pretrained styleGAN

set -eux

outputs="${STYLEGAN_DATA_DIR}/enriching_with_stylegan/outputs/inversion_val/"
data="${STYLEGAN_DATA_DIR}/enriching_with_stylegan/data/"
config="${STYLEGAN_CODE_DIR}/gan/configs/Set_Exemple/"
stats="${STYLEGAN_DATA_DIR}/enriching_with_stylegan/stats/"
weights="${STYLEGAN_DATA_DIR}/enriching_with_stylegan/weights/000024.pt"
pack="${STYLEGAN_DATA_DIR}/enriching_with_stylegan/outputs/pack_val/"

python3 main_inversion.py \
        --loss='mae' \
        --ckpt_dir=$weights \
        --stat_dir=$stats \
        --pack_dir=$pack \
        --real_data_dir=$data \
        --output_dir=$outputs \
        --dates_file="small_val.csv" \
        --date_start="2021-06-02" \
        --date_stop="2021-06-02"\
        --leadtimes=[3] \
        --invstep=1000