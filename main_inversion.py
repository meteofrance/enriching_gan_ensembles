#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:21:37 2023

@author: brochetc, moldovang

This script performs ensemble forecast inversion using a pre-trained StyleGAN2 model. 
The inversion process involves optimizing an initial random latent code so that it best represents a real ensemble forecast input.

The code uses command-line arguments for setting directories, inversion parameters, and data control parameters.
The inversion is performed for a specified set of dates and lead times, 
generating latent code representations for real-ensemble data and saving the results.

Please make sure to configure the directory paths, parameters, and other settings based on your specific environment before running the script.

"""
import argparse
import os
from collections import OrderedDict
from datetime import date, timedelta

import numpy as np
import pandas as pd
import torch

import perturbation.inversion as inv
import perturbation.utils as utils
from gan.model.stylegan2 import Generator

torch.manual_seed(42)  # reproducibility of runs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    ########################### Directories ###########################

    parser.add_argument("--ckpt_dir", type=str, default="./")
    parser.add_argument("--real_data_dir", type=str, default="./")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument(
        "--pack_dir", type=str, default="./"
    )  # storing "packed" (normalized) real data, grouped by ensemble in a single array

    parser.add_argument("--stat_dir", type=str, default="./")

    parser.add_argument("--device", type=str, default="cuda:0")
    ############################ INVERSION PARAMETERS #################

    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.05,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float,
        default=0.25,
        help="duration of the learning rate decay",
    )

    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")

    parser.add_argument(
        "--noise", type=float, default=0.005, help="strength of the noise level"
    )

    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )

    parser.add_argument("--invstep", type=int, default=1000, help="optimize iterations")
    parser.add_argument("--var_indices", type=utils.str2intlist, default=[1, 2, 3])
    parser.add_argument(
        "--Shape", type=tuple, default=(3, 256, 256), help="size of the samples"
    )

    parser.add_argument(
        "--noise_regularize",
        type=float,
        default=10e5,
        help="weight of the noise regularization (inversion)",
    )

    parser.add_argument(
        "--loss", type=str, default="mae", choices=["mse", "mae"]
    )
    parser.add_argument(
        "--loss_intens", type=float, default=1.0, help="weight of the pixel loss"
    )

    parser.add_argument(
        "--inv_checkpoints", type=utils.str2intlist, default=[200, 400, 600, 800, 1000]
    )

    ########################## CONTROL of Data to invert ######################

    parser.add_argument("--dates_file", type=str, default="val.csv")

    parser.add_argument("--date_start", type=str, default="2020-06-01")
    parser.add_argument("--date_stop", type=str, default="2021-11-15")
    parser.add_argument(
        "--leadtimes",
        type=utils.str2intlist,
        default=[3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45],
    )
    params = parser.parse_args()

    os.makedirs(params.pack_dir, exist_ok=True)
    os.makedirs(params.output_dir, exist_ok=True)

    # print(params.inv_checkpoints, type(params.inv_checkpoints[0]))
    assert type(params.inv_checkpoints[0]) == int
    ################## loading dates and file names ##

    df = pd.read_csv(params.real_data_dir + params.dates_file)

    df_date = df.copy()

    df_date["Date"] = pd.to_datetime(df_date["Date"])

    df_extract = df_date[
        (df_date["Date"] >= params.date_start) & (params.date_stop >= df_date["Date"] - timedelta(days=1))
    ]

    liste_dates = df_extract["Date"].unique()

    Means = np.load(f"{params.stat_dir}mean_rr_log.npy")[
        params.var_indices
    ].reshape(1, params.Shape[0], 1, 1)

    Maxs = np.load(f"{params.stat_dir}c_rr_log.npy")[
        params.var_indices
    ].reshape(1, params.Shape[0], 1, 1)

    ################ loading network #################

    device = params.device if torch.cuda.is_available() else "cpu"

    G = Generator(params.Shape[1], 512, n_mlp=8, nb_var=params.Shape[0])

    ckpt = torch.load(params.ckpt_dir, map_location="cpu")["g_ema"]

    if (
        "module" in list(ckpt.items())[0][0]
    ):  # juglling with Pytorch versioning and different module packaging
        ckpt_adapt = OrderedDict()
        for k in ckpt.keys():
            k0 = k[7:]
            ckpt_adapt[k0] = ckpt[k]
        G.load_state_dict(ckpt_adapt)

    else:
        G.load_state_dict(ckpt)
    G.eval()

    G = G.to(device)

    ################### producing latent mean #######

    if not os.path.exists(f"{params.output_dir}latent_mean.npy"):

        latent_z = torch.empty(10000, 512).normal_().to(device)
        with torch.no_grad():
            w = G.style(latent_z)

        latent_mean = w.mean(dim=0).detach().cpu()

        np.save(f"{params.output_dir}latent_mean.npy", latent_mean.numpy())
    else:

        lm = np.load(f"{params.output_dir}latent_mean.npy").astype(np.float32)
        latent_mean = torch.tensor(lm, dtype=torch.float32)

    #################### main loop ##################

    for date in liste_dates:
        datename = date.strftime("%Y-%m-%d")

        for lt in params.leadtimes:
            print(datename, lt)
            if not os.path.exists(
                params.output_dir + f"w_{datename}_{lt}_1000.npy"
            ):  # cecking for already teer

                Ens_r = utils.load_batch_from_timestamp(
                    df_extract,
                    date,
                    lt - 1,
                    params.real_data_dir,
                    var_indices=params.var_indices,
                )

                Ens_r = torch.tensor(0.95 * (Ens_r - Means) / Maxs, dtype=torch.float32)

                np.save(
                    params.pack_dir + f"Rsemble_{datename}_{lt}.npy",
                    Ens_r.numpy().astype(np.float32),
                )

                params.date_index = datename
                params.lt_index = lt

                inv.optimize(Ens_r, G, latent_mean, device, params)
