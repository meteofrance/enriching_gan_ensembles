#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:42:52 2022

@author: brochetc

plot length scales

"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid

dir0 = "/home/brochetc/Bureau/These/presentations_these/Deuxieme annee/"  # change to your path

real_corr = np.load(dir0 + "real_corr_length.npy")
corr_length = np.load(dir0 + "fake_corr_length.npy")

fig = plt.figure(figsize=(6, 9))

grid = ImageGrid(
    fig,
    111,
    nrows_ncols=(2, 3),
    axes_pad=(0.35, 0.35),
    label_mode="L",
    share_all=True,
    cbar_location="bottom",
    cbar_mode="edge",
    cbar_size="10%",
    cbar_pad="20%",
)
ax4 = grid[3]
im4 = ax4.imshow(corr_length[-1, 0, :, :], origin="lower")
cb4 = ax4.cax.colorbar(im4)


ax5 = grid[4]
im5 = ax5.imshow(corr_length[-1, 1, :, :], origin="lower")
cb5 = ax5.cax.colorbar(im5)

ax6 = grid[5]
im6 = ax6.imshow(corr_length[-1, 2, :, :], origin="lower")
cb6 = ax6.cax.colorbar(im6)

ax = grid[0]
ax.imshow(
    real_corr[0, :, :],
    origin="lower",
    vmin=corr_length[-1, 0].min(),
    vmax=corr_length[-1, 0].max(),
)

ax2 = grid[1]
ax2.imshow(
    real_corr[1, :, :],
    origin="lower",
    vmin=corr_length[-1, 1].min(),
    vmax=corr_length[-1, 1].max(),
)

ax3 = grid[2]
ax3.imshow(
    real_corr[2, :, :],
    origin="lower",
    vmin=corr_length[-1, 2].min(),
    vmax=corr_length[-1, 2].max(),
)
