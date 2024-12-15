#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 11:42:53 2022

@author: brochetc

multivariate_plot

"""

import pickle

import matplotlib.pyplot as plt
import multivariate as mlt
import numpy as np

path = "/home/brochetc/Bureau/These/presentations_these/Premiere_annee/ressources_presentations/multivariables/Set_38_32_2/"  # change to your path

if __name__ == "__main__":

    res = pickle.load(open(path + "multivar_1_distance_metrics_16384.p", "rb"))

    RES = res["multivar"].squeeze()

    for i in range(41):
        print(i)
        data_r, data_f = RES[i, 0], RES[i, 1]
        print(data_r.shape, data_f.shape)

        levels = mlt.define_levels(data_r, 5)
        ncouples2 = data_f.shape[0] * (data_f.shape[0] - 1)
        bins = np.linspace(
            tuple([-1 for i in range(ncouples2)]),
            tuple([1 for i in range(ncouples2)]),
            101,
            axis=1,
        )

        var_r = (np.log(data_r), bins)
        var_f = (np.log(data_f), bins)

        mlt.plot2D_histo(
            var_f, var_r, levels, output_dir=path, add_name="check_order_{}".format(i)
        )
