#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 17:26:05 2021

@author: brochetc
"""

from time import sleep

import residual_nets as ResN
import torch
import torch.cuda as cu
import torch.optim as optim
from torch.nn.functional import relu


def get_memory_footprint(model):
    """
    copied from @ptrblck, NVIDIA
    output is memory footprint in bytes/octet
    """
    mem_params = sum(
        [param.nelement() * param.element_size() for param in model.parameters()]
    )
    mem_buff = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    return mem_params, mem_buff, mem_params + mem_buff


def memoryInGB():
    return cu.memory_allocated() / (1e9)


def toTheSky(input_dim, nz, input_channels, n0):

    n = n0

    while n < 100:
        print(2**n)
        try:
            batch_size = 2**n

            modelD = ResN.ResNet_D(input_dim, input_channels).cuda()
            modelG = ResN.ResNet_G(nz, input_dim, input_channels).cuda()
            optim_D = optim.Adam(modelD.parameters(), lr=1e-4)
            optim_G = optim.Adam(modelG.parameters(), lr=1e-4)
            print("models and optimizer loading, cost  {:.4f} GB".format(memoryInGB()))
            sleep(1)

            X = torch.randn(batch_size, input_channels, 128, 128).cuda()
            print("sample, cost  {:.4f} GB".format(memoryInGB()))
            sleep(1)
            pred_X = modelD(X)
            print("prediction, cost  {:.4f} GB".format(memoryInGB()))
            sleep(1)

            Z = torch.randn(batch_size, nz).cuda()
            print("sample Z, cost  {:.4f} GB".format(memoryInGB()))
            sleep(1)

            pred_Z = modelD(modelG(Z))
            print("prediction, cost  {:.4f} GB".format(memoryInGB()))
            sleep(1)

            loss = relu(1.0 - pred_X).mean() + relu(1.0 + pred_Z).mean()
            optim_D.zero_grad()
            loss.backward()
            optim_D.step()
            print("loss step, cost  {:.4f} GB".format(memoryInGB()))
            sleep(1)

            Z = torch.randn(batch_size, nz).cuda()
            X_Z = modelG(Z)
            print("sample and pred Z, cost  {:.4f} GB".format(memoryInGB()))
            sleep(1)

            loss = -modelD(X_Z).mean()
            optim_G.zero_grad()
            loss.backward()
            optim_G.step()
            print("loss step, cost  {:.4f} GB".format(memoryInGB()))
            sleep(1)

        except RuntimeError:
            print(
                "{} is the max batch size possible\
                 with those architectures".format(
                    2**n
                )
            )
            break
        cu.memory_summary()
        ###freeing memory ###
        del loss, Z, X_Z, X, modelD, modelG, optim_G, optim_D
        torch.cuda.empty_cache()
        cu.memory_summary()

        n += 1
    torch.cuda.empty_cache()
    return n


if __name__ == "__main__":

    input_Dims = [32, 64, 128]
    NZs = [32, 64, 128, 256]
    n = 1
    for input_dim in input_Dims:

        for nz in NZs:
            print(
                "testing architectures with nz= {}, input_dim = {}".format(
                    nz, input_dim
                )
            )

            n = toTheSky(
                input_dim, nz, 4, n
            )  # old n is new start as complexity increases
