#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:58:44 2023

@author: brochetc

Stats on Masked files

"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

base_path = "/cnrm/recyf/NO_SAVE/Data/users/brochetc/float32_t2m_u_v_rr/full_domain/IS_1_1.0_0_0_0_0_0_1024_trial/"


def computeUnderMask(data, MaskV, func):
    """
    Data must be of shape C x H x W
    Computation is performed only with unitary functions

    MaskV is a mask fill value
    """
    data_to_comp = np.ma.masked_values(data, MaskV)

    quant = func(data_to_comp)

    return quant


def normalizeUnderMask(data, MaskV, Mean, std, scale, fill=0.0):
    """
    Data must be of shape C x H x W

    Mean is array_like (can be scalar), should be broadcastable to shape C x 1 x 1, idem std

    scale is a float

    MaskV is a threshold mask value (typically 9999.0)

    Returns normalized data masked values left untouched

    """
    data_to_norm = np.ma.masked_values(data, MaskV)

    data_norm_mask = scale * (data_to_norm - Mean) / std

    return np.ma.filled(data_norm_mask, fill_value=fill)


def normalizeDataset(data, path, var_indices=list(range(1, 4)), verbose=False):
    """
    Parameters :

        data : array to be normalized
        path : where to find constants
        var_indices : array-like containing the indices of the variables of interest

    Returns :

        normalized data such that :

            - mean of the WHOLE dataset is (approx.) 0.0 for each variable
            - max OR min of the WHOLE dataset is (approx.) 0.95 OR -0.95 for
            each variable

    """

    Means = np.load(path + "Mean_4_var.npy")[var_indices].reshape(
        len(var_indices), 1, 1
    )
    Mins = np.load(path + "Min_4_var.npy")[var_indices].reshape(len(var_indices), 1, 1)
    Maxs = np.load(path + "Max_4_var.npy")[var_indices].reshape(len(var_indices), 1, 1)

    MinC, MaxC = np.abs(Mins - Means), np.abs(Maxs - Means)

    Div = (MaxC + MinC) / 2.0 + np.abs(MaxC - MinC) / 2.0
    # fancy way to compute the max of MinC, MaxC

    # fill with 0.0 is performed AFTER normalization
    normed = normalizeUnderMask(data, 9999.0, Means, Div, 0.95, fill=0.0)

    if verbose:

        funcs = {
            "Mean": lambda m: np.mean(m, axis=(0, -2, -1)),
            "Min": lambda m: np.min(m, axis=(0, -2, -1)),
            "Max": lambda m: np.max(m, axis=(0, -2, -1)),
        }

        print(
            computeUnderMask(normed, 0.0, funcs["Mean"]),
            computeUnderMask(normed, 0.0, funcs["Min"]),
            computeUnderMask(normed, 0.0, funcs["Max"]),
        )

    return normed


if __name__ == "__main__":

    var_names = ["u", "v", "t2m"]
    data = np.zeros((128, 3, 1024, 1024))
    print("collecting 128 samples")
    for i in range(128):

        data[i] = np.load(base_path + "_sample{}.npy".format(i))

    for j in range(3):

        cmap = "coolwarm" if j == 2 else "viridis"

        plt.imshow(data[0, j], origin="lower")
        plt.colorbar()
        plt.savefig("true_data_norm_test_{}.png".format(var_names[i]))
        plt.close()
    print("normalizing")
    Normed = normalizeDataset(data, base_path, verbose=True)

    for j in range(3):

        cmap = "coolwarm" if j == 2 else "viridis"

        plt.imshow(Normed[0, j], origin="lower")
        plt.colorbar()
        plt.savefig("norm_data_norm_test_{}.png".format(var_names[i]))
        plt.close()

    print("done, tests plotted")


def collectAndCompute(base_path=base_path):

    funcs = {
        "Mean": lambda m: np.mean(m, axis=(-2, -1)),
        "Min": lambda m: np.min(m, axis=(-2, -1)),
        "Max": lambda m: np.max(m, axis=(-2, -1)),
    }

    print("globbing")
    filelist = glob.glob(base_path + "_sample*.npy")
    # print([filelist[i] for i in range(0,66048,1024)])
    # print([f[-17:] for f in filelist[-515:]])

    Sum = np.array(
        [15390.530330379323, 62831.02876115003, 13717.80083381357, 18799140.00650119]
    )

    gMin = np.array([0.0, -34.22730255126953, -35.41071319580078, 240.27041625976562])

    gMax = np.array(
        [574.4132080078125, 40.038841247558594, 36.08406066894531, 321.4552001953125]
    )

    print(filelist[-512])
    for i, f in enumerate(filelist[-511:]):

        print(f[-16:])

        data = np.load(f).astype(np.float32)

        Avg = computeUnderMask(data, 9999.0, funcs["Mean"])
        Min = computeUnderMask(data, 9999.0, funcs["Min"])
        Max = computeUnderMask(data, 9999.0, funcs["Max"])

        """if i==0 :
            gMin = Min
            gMax = Max"""

        Sum = Sum + Avg

        gMin = (Min + gMin) / 2.0 - np.abs(Min - gMin) / 2.0
        gMax = (Max + gMax) / 2.0 + np.abs(Max - gMax) / 2.0

        print(Avg, Min, Max)
        print(Sum, gMin, gMax)
        print("###########################")

    Sum = Sum / len(filelist)
    np.save(base_path + "Mean_4_var.npy", Sum.__array__())
    np.save(base_path + "Min_4_var.npy", gMax.__array__())
    np.save(base_path + "Max_4_var.npy", gMin.__array__())
