#!/usr/bin/bash

# run an inversion on a given set with a pretrained styleGAN

set -eux

outputs="${STYLEGAN_DATA_DIR}/enriching_with_stylegan/outputs/perturbation_val/"
data="${STYLEGAN_DATA_DIR}/enriching_with_stylegan/data/"
invdata="${STYLEGAN_DATA_DIR}/enriching_with_stylegan/outputs/inversion_val/"
config="${STYLEGAN_CODE_DIR}/gan/configs/Set_Exemple/"
stats="${STYLEGAN_DATA_DIR}/enriching_with_stylegan/stats/"
ckpt="${STYLEGAN_DATA_DIR}/enriching_with_stylegan/weights/000024.pt"
pack="${STYLEGAN_DATA_DIR}/enriching_with_stylegan/outputs/pack_val/"
weights="${STYLEGAN_DATA_DIR}/enriching_with_stylegan/weights/"

python3 main_perturbation.py \
        --output_dir=$outputs \
        --real_data_dir=$data \
        --eigen_dir=$weights \
        --data_dir=$invdata \
        --stat_dir=$stats \
        --pack_dir=$pack \
        --ckpt_dir=$ckpt \
        --N_samples=112 \
        --N_seeds=16 \
        --N_conditioners=16 \
        --style_indices=[1,1,1,1,1,1,1,1,1,1,0,0,0,0] \
        --beta_dir=$weights \
        --dates_file='small_val.csv' \
        --add_name='this_experiment' \
        --date_start="2021-06-02" \
        --date_stop="2021-06-02" \
        --leadtimes=[3] \
        --invstep=200 \
        --verbose