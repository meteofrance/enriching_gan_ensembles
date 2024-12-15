#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:13:57 2023

@author: brochetc

The aim is to analyze the effect of intra-ensemble latent uncertainty
on the real space variability

"""
import torch
import torch.linalg as linalg


def ensemble_pod(Ens, cut, verbose=False):
    """
    Compute the first (= highest eigenvalues) eigenvectors of the Ens covariance matrix

    Ens : B x N x D torch.tensor where N is the number of samples and D the dimension
    of the space where they live

    cut : int, additional parameter to set all eigenvalues with rank > cut to zero.

    Return:

        eigenvalues :  torch.tensor ; eigenvalues (absolute values) of the ensemble Ens up to cut (and zero after cut)
            shape B x D
        eigenvectors :  torch.tensor ; eigenvectors associated to d. Shape B x D x D
    """

    if verbose:
        print(Ens.shape)

    size = Ens.shape[1] if Ens.ndim == 3 else Ens.shape[0]
    if verbose:
        print("size", size)

    Dim = Ens.shape[-1]

    if Ens.ndim == 3:

        Ens_t = Ens.permute((0, 2, 1))

    else:

        Ens_t = Ens.t().unsqueeze(0)
        Ens = Ens.unsqueeze(0)

    if verbose:
        print("(transpose) ensemble shape", Ens_t.shape, Ens.shape)

    if size > 1:
        cov_matrix = torch.bmm(Ens_t, Ens) * (1 / (size - 1))
    else:
        cov_matrix = torch.bmm(Ens_t, Ens)

    if verbose:
        print("empirical cov shape", cov_matrix.shape)

    eigenvalues, eigenvectors = linalg.eigh(cov_matrix)

    if verbose:
        print("remaining values on 0th index", eigenvalues[0, Dim - cut :])

    eigenvalues[:, : Dim - cut] = torch.zeros_like(eigenvalues[:, : Dim - cut])

    if verbose:
        print("max diag", eigenvalues.max())

    return eigenvalues, eigenvectors


def compute_K_covariance(Ens_w, cut, verbose=False, device="cuda:0"):
    N, R, D = Ens_w.shape  # Number, Repeats, Dimension (typicallly = 16, 14, 512)

    # Ens_w1 = Ens_w * torch.rsqrt(EnsNorm + 1e-8)
    w_avg = Ens_w.mean(dim=0)

    if verbose:
        print("Ensemble w shape", Ens_w.shape, w_avg.shape)

    Ens_0 = (Ens_w - w_avg).view(R, N, D)

    sigmas, q = ensemble_pod(Ens_0, cut=cut, verbose=verbose)
    if verbose:
        print("Sigmas and q shape", sigmas.shape, q.shape)
    sigmas = torch.sqrt(torch.abs(sigmas))
    sigmas = torch.diag_embed(sigmas)

    if verbose:
        print("Max, min values, q shape", sigmas.max(), sigmas.min(), q.shape)

    assert torch.isfinite(q).all()
    if verbose:
        id_test = torch.bmm(q, q.permute(0, 2, 1)) - torch.eye(q.shape[-1]).to(device)
        print(
            "id_test linalg norm",
            torch.linalg.norm(id_test),
            torch.max(torch.abs(id_test)),
        )

    K = torch.bmm(
        q, torch.bmm(sigmas, q.permute(0, 2, 1))
    ).contiguous()  # shape R x D x D

    return K, w_avg
