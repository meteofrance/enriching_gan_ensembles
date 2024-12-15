#!/usr/bin/bash

# run an inversion on a given set with a pretrained styleGAN

set -eux

outputs="${STYLEGAN_DATA_DIR}/enriching_with_stylegan/outputs/scale_tune_val/"
data="${STYLEGAN_DATA_DIR}/enriching_with_stylegan/data/"
invdata="${STYLEGAN_DATA_DIR}/enriching_with_stylegan/outputs/inversion_val/"
config="${STYLEGAN_CODE_DIR}/gan/configs/Set_Exemple/"
stats="${STYLEGAN_DATA_DIR}/enriching_with_stylegan/stats/"
ckpt="${STYLEGAN_DATA_DIR}/enriching_with_stylegan/weights/000024.pt"
pack="${STYLEGAN_DATA_DIR}/enriching_with_stylegan/outputs/pack_val/"
weights="${STYLEGAN_DATA_DIR}/enriching_with_stylegan/weights/"

python3 scale_tune.py \
        --output_dir=$outputs \
        --fake_data_dir=$invdata \
        --real_data_dir=$data \
        --eigen_dir=$weights \
        --pack_dir=$pack \
        --ckpt_dir=$ckpt \
        --inflate=0.2 \
        --style_cut=10 \
        --n_samples=16 \
        --dates_file='small_val.csv' \
        --date_start="2021-06-02" \
        --date_stop="2021-06-02"\
        --invstep=200 \
        --leadtimes=[3]\
        --n_epochs=50