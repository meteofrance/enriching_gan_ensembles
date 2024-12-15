#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:21:37 2023

@author: brochetc

Main pod sampling script

"""

import argparse
import os
import pickle
import random
from collections import OrderedDict
from datetime import date, datetime, timedelta
from shutil import copyfile
from time import perf_counter

import numpy as np
import pandas as pd
import torch

import perturbation.smpca as smpca
import perturbation.utils as utils
from gan.model.stylegan2 import Generator

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def str2list(li):
    if type(li) == list:
        li2 = li
        return li2

    elif type(li) == str:
        li2 = li[1:-1].split(",")
        return li2

    else:
        raise ValueError(
            "li argument must be a string or a list, not '{}'".format(type(li))
        )


def compute_generate_save(G, params):
    """generate

    Args:
        G (nn.Module): stylegan generator
        params (Namespace): parameters Namespace (output by args)
    """

    N_samples = params.N_samples
    N_seeds = params.N_seeds
    if params.verbose:
        print(datename, lt)
        print(params.date_index, params.lt_index)
    Ens_r = torch.tensor(
        np.load(params.pack_dir + f"Rsemble_{datename}_{lt}.npy"), dtype=torch.float32
    )
    w_ens = torch.tensor(
        np.load(
            params.data_dir
            + f"w_{params.date_index}_{params.lt_index}_{params.invstep}.npy"
        ).astype(np.float32)
    )

    # subsampling if N_conditioners is lower than initial ensemble size
    if params.N_conditioners < w_ens.shape[0]:
        cond_indices = random.sample(range(w_ens.shape[0]), params.N_conditioners)
        w_ens = w_ens[cond_indices]
        Ens_r = Ens_r[cond_indices]
    print("############### Perturbating ###############")
    print("loading generation hyperparams")
    Whitening = (
        torch.load(params.eigen_dir + "Whitening.pt")
    )

    w0 = (
        torch.load(params.eigen_dir + "latent_mean.pt")
    )
    betas = torch.tensor(
        np.load(os.path.join(params.beta_dir, "ema_betas.npy")).astype(np.float32)[
            params.betas_alphas_step
        ],
        device=device,
    )
    alphas = torch.tensor(
        np.load(os.path.join(params.beta_dir, "ema_alphas.npy")).astype(np.float32)[
            params.betas_alphas_step
        ],
        device=device,
    )

    gen, _ = smpca.sm_pca(
        w_ens,
        G,
        N_samples,
        params.style_indices,
        N_seeds,
        betas,
        alphas,
        Whitening=Whitening,
        w0=w0,
        device=device,
        verbose=params.verbose,
    )

    if params.verbose:
        print(gen.mean(axis=(0, -2, -1)))
        print(Ens_r.mean(axis=(0, -2, -1)))

    # saving after generation
    np.save(
        params.output_dir
        + f"genFsemble_{params.date_index}_{params.lt_index}_{params.invstep}_{params.N_conditioners}.npy",
        gen,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    ########################### Directories ###########################
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./",
    )
    parser.add_argument(
        "--real_data_dir",
        type=str,
        default="./",
    )
    parser.add_argument(
        "--stat_dir",
        type=str,
        default="./",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
    )
    parser.add_argument("--add_name", type=str, default="")
    parser.add_argument(
        "--eigen_dir",
        type=str,
        default="./",
    )
    parser.add_argument(
        "--pack_dir", type=str, default="./"
    )  # storing "packed" (normalized) real data

    parser.add_argument("--mean_file", type=str, default="Mean_4_var.npy")
    parser.add_argument("--max_file", type=str, default="MaxNew_4_var.npy")

    parser.add_argument("--var_indices", type=utils.str2intlist, default=[1, 2, 3])
    parser.add_argument(
        "--Shape", type=tuple, default=(3, 256, 256), help="size of the samples"
    )
    parser.add_argument(
        "--N_samples", type=int, default=120, help="number of new samples"
    )
    parser.add_argument(
        "--N_conditioners",
        type=int,
        default=16,
        help="number of 'seed' samples used for estimating the gain matrices",
    )
    parser.add_argument(
        "--N_seeds", type=int, default=16, help="number of seeds members to perturbate"
    )
    parser.add_argument(
        "--invstep", type=int, default=1000, help="step of inversion to load w"
    )
    ######################## PERTURBATION PARAMETERS #######################

    parser.add_argument(
        "--style_indices", type=str2list, default="[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]"
    )
    parser.add_argument("--beta_dir", type=str, default="./")
    parser.add_argument("--betas_alphas_step", type=int, default=-1)

    ########################## CONTROL of Data to perturb ######################

    parser.add_argument("--dates_file", type=str, default="Large_lt_val_labels.csv")

    parser.add_argument("--date_start", type=str, default="2020-07-01")
    parser.add_argument("--date_stop", type=str, default="2020-12-31")
    parser.add_argument(
        "--leadtimes",
        type=utils.str2intlist,
        default=[3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45],
    )

    ###########################################################################
    parser.add_argument("--runtime_metrics", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    ###########################################################################

    params = parser.parse_args()
   
    root_dir = params.output_dir
    params.output_dir = (
        params.output_dir
        + f"{params.style_indices}_{params.betas_alphas_step}_{params.N_conditioners}_{params.N_seeds}_{params.add_name}/"
    )
    os.makedirs(params.output_dir,exist_ok=True)
    ################## selecting dates
    print("reading dates")
    df = pd.read_csv(params.real_data_dir + params.dates_file)
    df_date = df.copy()
    df_date["Date"] = pd.to_datetime(df_date["Date"])
    df_extract = df_date[
        (df_date["Date"] >= params.date_start) &  (params.date_stop >= df_date["Date"] - timedelta(days=1))
    ]
    liste_dates = df_extract["Date"].unique()

    ################## carrying scaling info to pass it whenever needed
    Means = np.load(f"{params.stat_dir}mean_rr_log.npy")[
        params.var_indices
    ].reshape(1, params.Shape[0], 1, 1)
    Maxs = np.load(f"{params.stat_dir}c_rr_log.npy")[
        params.var_indices
    ].reshape(1, params.Shape[0], 1, 1)
    scale = 1 / 0.95


    ################ loading network #################

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("loading G")
    G = Generator(params.Shape[1], 512, n_mlp=8, nb_var=params.Shape[0])
    ckpt = torch.load(params.ckpt_dir, map_location="cpu")["g_ema"]
    if (
        "module" in list(ckpt.items())[0][0]
    ):  # juggling with Pytorch versioning and different module packaging
        ckpt_adapt = OrderedDict()
        for k in ckpt.keys():
            k0 = k[7:]
            ckpt_adapt[k0] = ckpt[k]
        G.load_state_dict(ckpt_adapt)
    else:
        G.load_state_dict(ckpt)
    G.eval()
    G = G.to(device)

    #############################  Main loop ###############################

    metrics_list = ["variance", "std_diff"]  # , 'mean_bias']
    metrics = {}
    for date in liste_dates:
        datename = date.strftime("%Y-%m-%d")
        for lt in params.leadtimes:
            print(datename, lt)
            params.date_index = datename
            params.lt_index = lt
            try:
                print("generating")
                compute_generate_save(
                    G, params
                )
            except FileNotFoundError as e:
                print(f"File not found {e}")
                pass