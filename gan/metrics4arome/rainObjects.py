#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 13:37:28 2023

@author: brochetc
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as sc

gaussian_filter = (
    np.float32(
        [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ]
    )
    / 256.0
)


def blurFunc(data, filter_conv=None):

    data_filter = sc.convolve(
        data, filter_conv[np.newaxis, np.newaxis, :, :], mode="mirror"
    )

    return data_filter


def threshold_and_gather(data, data_filtered, threshold):

    mask = (data_filtered > threshold).squeeze().astype(np.int8)

    interesting = data * mask

    return interesting[interesting > 0], mask


def threshold_and_gather_2ends(data, data_filtered, t1, t2):

    mask1 = (data_filtered > t1).squeeze().astype(np.int8)

    mask2 = (data_filtered < t2).squeeze().astype(np.int8)

    mask = np.abs(np.minimum(np.zeros_like(mask1), 1 - (mask1 + mask2))).astype(np.int8)

    interesting = data * mask

    return interesting[interesting > 0], mask


if __name__ == "__main__":

    data_dir = "./"
    sample_name = "_sample1059.npy"

    data_rr = np.load(data_dir + sample_name).astype(np.float32)[0:1]

    data = np.expand_dims(data_rr, axis=0)

    data_filtered = blurFunc(data, gaussian_filter)  # un peu de floutage pour lisser

    plt.imshow(
        data_filtered[0, 0], origin="lower", cmap="Blues"
    )  # visualisation du flou
    plt.show()

    THRS = [
        0.1,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
        5.5,
        6.0,
        6.5,
        7.0,
        7.5,
        8.0,
    ]

    THRS1 = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    THRS2 = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    areas = []

    q90 = []

    q95 = []

    q99 = []

    q50 = []

    for t1 in THRS:

        objects, mask = threshold_and_gather(data, data_filtered, t1)
        # on définit un masque où se trouvent les 'objets' et on peut
        # regarder comment se comportent 1) la distribution des points à l'intérieur
        # et 2) la taille de l'objet

        # plt.close()

        # plt.hist(np.log(1 + objects), bins=50)
        # plt.show()
        # plt.close()

        plt.imshow(mask, origin="lower")
        plt.show()

        areas.append(mask.sum() * 2.6**2)  # aire de " l'objet" en km²

        q90.append(np.quantile(objects, 0.9))

        q95.append(np.quantile(objects, 0.95))
        q50.append(np.quantile(objects, 0.5))
        q99.append(np.quantile(objects, 0.99))

    plt.plot(THRS, np.log10(areas))
    plt.show()

    plt.plot(THRS, q90)
    plt.plot(THRS, q95)
    plt.plot(THRS, q99)

    plt.plot(THRS, q50)
    plt.show()
