#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:04:50 2023

@author: brochetc

Style-Mixed PCA
"""
import random
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F

import perturbation.pca_stylegan as pca


def sm_pca(
    # initial (inverted) ensemble in latent space
    # shape is Nens x R (=14 for 256x256 images) x D (=512 in our case)
    Ens_w,
    # stylegan generator network
    G,
    # number of desired samples
    N_samples,
    # which styles to perturb : 0 = randomly, 1 = with K ; length
    sm_ind=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # number of seeds
    N_seeds=16,
    # intensity of stochastic fluctuations ; can be either int or array of shape R
    betas=1.0,
    # strength of the ensemble average ; can be either int or array of shape R
    alphas=0.0,
    # "whitening" = decorrelating matrix (making vectors "white noise")
    Whitening=None,
    # average vector of W latent space
    w0=None,
    # shape of an individual tensor
    shape=(3, 256, 256),
    # device on which generation appears
    device="cuda:0",
    verbose=False,
):

    N, R, D = Ens_w.shape
    # the number of new samples per seed
    per_cond = int(ceil(N_samples / N_seeds))

    # final placeholder for generated samples (filled in progressively)
    Ens_final = np.zeros((N * per_cond, *shape), dtype="float32")
    w_final = np.zeros((N * per_cond, R, D))

    sm_ind_np = np.array(sm_ind).astype(np.bool_)

    # extracting the styles from Ens_w which will be perturbed with K
    w_K = Ens_w[:, sm_ind_np, :].to(device)
    if verbose:
        print(f"Extracted w_extract {w_K.size()}")
    n_styles_pert = w_K.size()[1]

    assert Whitening is not None
    assert w0 is not None
    if verbose:
        print(f"betas (scale factor) {betas}")

    # computing K for each style in w_K
    if n_styles_pert > 0:
        # shape n_styles_pert x D x D
        K, _ = pca.compute_K_covariance(w_K, cut=N - 1, verbose=verbose, device=device)

    else:
        K = None

    Ens_w1 = Ens_w.to(device)
    if N_seeds < N:
        # if there are less seeds than ensemble members, perform a random sampling of the ensembles
        seeds = random.sample(range(N), N_seeds)
        Ens_w1 = Ens_w[seeds].to(device)
        print(Ens_w1.shape)

    with torch.no_grad():
        # generating a common multiple of each conditioning sample
        for k in range(N_seeds):
            if verbose:
                print(f"member {k} is fixed")

            # fluctuations with K
            if n_styles_pert:
                z = torch.empty((per_cond, D)).normal_().contiguous().to(device)
                with torch.no_grad():
                    w = G.style(z)

                # matrix multiplication to make newly sampled vectors "uncorrelated"
                diff = torch.bmm(
                    Whitening.to(device).unsqueeze(0).repeat(per_cond, 1, 1),
                    (w - w.mean(dim=0)).unsqueeze(-1),
                )  # diff of shape N_samples  x D
                w_stoch = torch.einsum("abc, dc-> dab", K, diff.squeeze(dim=-1))

            # interpolating between ensemble mean and individual sample
            w_start = (
                alphas.view(1, R, 1) * Ens_w1.mean(dim=0)
                + (1.0 - alphas).view(1, R, 1) * Ens_w1[k]
            )

            # creating random perturbations if needed and concatenating on finest styles
            if (R - n_styles_pert) > 0:
                z = torch.empty((per_cond, 512)).normal_().to(device)
                with torch.no_grad():
                    w_rdm = G.style(z)
                if n_styles_pert > 0:
                    w_pert = torch.cat(
                        [
                            w_stoch,
                            (w_rdm - w_rdm.mean(dim=0))
                            .unsqueeze(1)
                            .repeat(1, (R - n_styles_pert), 1),
                        ],
                        dim=1,
                    )
                else:
                    w_pert = (
                        (w_rdm - w_rdm.mean(dim=0))
                        .unsqueeze(1)
                        .repeat(1, (R - n_styles_pert), 1)
                    )
            else:
                w_pert = w_stoch

            # main formula for vector perturbation
            w_new = w_start + betas.view(1, 14, 1) * w_pert

            assert torch.isfinite(w_new).all()
            if verbose:
                print("wnew", w_new.shape)
            sample, _, _ = G([w_new.to(device)], input_is_latent=True)
            Ens_final[k * per_cond : (k + 1) * per_cond] = sample.detach().cpu().numpy()
            w_final[k * per_cond : (k + 1) * per_cond] = w_new.detach().cpu().numpy()

    return Ens_final[:N_samples], w_final


def fast_style_mixing(
    alphas,
    betas,
    batch_w,
    K,
    w_avg,
    G,
    Whitening,
    device="cpu",
    beta_rule="linear",
):
    """
    Perform style mixing using interpolation coefficients (alpha's) and scale coefficients (beta's)
    and make the resulting physical samples differentiable wrt alpha's and beta's
    To be used with scale_tune script (faster)
    """
    R, D = w_avg.shape  # Repeats, Dimension (typicallly = 14, 512)
    n_styles_no_pca = 14 - R
    n_samples = batch_w.shape[0]
    # sigmoid is applied to alphas --> stored value is thus sigmoid(alphas)
    w_start = (
        F.sigmoid(alphas).view(1, 14, 1) * batch_w.mean(dim=0)
        + (1.0 - F.sigmoid(alphas).view(1, 14, 1)) * batch_w
    )

    # perturbation on styles with K
    if R > 0:
        z = torch.empty((n_samples, D)).normal_().contiguous().to(device)
        with torch.no_grad():
            w = G.style(z)
        diff = torch.bmm(
            Whitening.to(device).unsqueeze(0).repeat(n_samples, 1, 1),
            (w - w.mean(dim=0)).unsqueeze(-1),
        )  # diff of shape N_samples  x D
        new_w = torch.einsum("abc, dc-> dab", K, diff.squeeze(dim=-1))

    # perturbation on styles with random latents
    if n_styles_no_pca > 0:
        z = torch.empty((n_samples, 512)).normal_().to(device)
        with torch.no_grad():
            w_nopca = G.style(z)
        if R > 0:
            w_pert = torch.cat(
                [
                    new_w,
                    (w_nopca - w_nopca.mean(dim=0))
                    .unsqueeze(1)
                    .repeat(1, n_styles_no_pca, 1),
                ],
                dim=1,
            )
        else:
            w_pert = (
                (w_nopca - w_nopca.mean(dim=0))
                .unsqueeze(1)
                .repeat(1, n_styles_no_pca, 1)
            )
    else:
        w_pert = new_w

    # betas viewed as linear parameters
    if beta_rule == "linear":
        res = w_start + betas.view(1, 14, 1) * w_pert
        gen, _, _ = G([res], input_is_latent=True)

    # constraining betas to be strictly in (0,1)
    elif beta_rule == "sigmoid":
        res = w_start + F.sigmoid(betas).view(1, 14, 1) * w_pert
    gen, _, _ = G([res], input_is_latent=True)

    return gen
