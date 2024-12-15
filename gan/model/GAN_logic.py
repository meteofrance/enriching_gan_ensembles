#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:43:43 2022

@author: brochetc
"""
import math
import random
from functools import wraps
from time import perf_counter

import torch
import torch.nn.functional as F
from torch import autograd

from gan.model.op import conv2d_gradfix


def timer(func):
    @wraps(func)
    def time_measurement(*args, **kwargs):
        t0 = perf_counter()
        res = func(*args, **kwargs)
        t1 = perf_counter() - t0
        print(f"Function {func.__name__} Took {t1:.4f} seconds")
        return res

    return time_measurement


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def make_noise(batch, latent_dim, n_noise):
    if n_noise == 1:
        return torch.randn(batch, latent_dim).cuda()

    noises = torch.randn(n_noise, batch, latent_dim).cuda().unbind(0)

    return noises


# typical styleGAN funcs


def mixing_noise(batch, latent_dim, prob):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2)

    else:
        return [make_noise(batch, latent_dim, 1)]


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):

    with conv2d_gradfix.no_weight_gradients():
        (grad_real,) = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):

    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )

    noise = noise.cuda()

    grad = autograd.grad(
        outputs=(fake_img * noise).sum(),
        inputs=latents,
        create_graph=True,
        only_inputs=True,
    )[0]

    path_lengths = grad.square().sum(2).mean(1).sqrt()

    path_mean = torch.lerp(mean_path_length.cuda(), path_lengths.mean(), decay)

    path_penalty = (path_lengths - path_mean).square().mean()

    return path_penalty, path_mean.detach(), path_lengths


# TODO : augmentation implementation
# TODO : conditional setup


def Discrim_Step_StyleGAN(
    samples,
    modelD,
    modelG,
    latent_dim,
    mixing=0,
):
    loss_dict = {}

    requires_grad(modelG, False)
    requires_grad(modelD, True)

    noise = mixing_noise(samples.shape[0], latent_dim, mixing)
    with torch.no_grad():

        fake, _, _ = modelG(noise)

    fake_pred = modelD(fake)
    real_pred = modelD(samples)

    d_loss = d_logistic_loss(real_pred, fake_pred)

    loss_dict["d"] = d_loss
    loss_dict["real_score"] = real_pred.mean()
    loss_dict["fake_score"] = fake_pred.mean()

    for param in modelD.parameters():
        param.grad = None
    d_loss.backward()

    return loss_dict


def Discrim_Regularize(samples, modelD, r1, d_reg_every):

    loss_dict = {}

    samples.requires_grad = True

    real_pred = modelD(samples)
    r1_loss = d_r1_loss(real_pred, samples)

    for param in modelD.parameters():
        param.grad = None

    (r1 / 2 * r1_loss * d_reg_every + 0 * real_pred[0]).backward()

    loss_dict["r1"] = r1_loss

    return loss_dict


def Generator_Step_StyleGAN(
    samples,
    modelD,
    modelG,
    latent_dim,
    mixing=0,
):

    loss_dict = {}

    for param in modelG.parameters():
        param.grad = None

    noise = mixing_noise(samples.shape[0], latent_dim, mixing)

    fake, _, _ = modelG(noise)

    fake_pred = modelD(fake)
    g_loss = g_nonsaturating_loss(fake_pred)

    loss_dict["g"] = g_loss

    g_loss.backward()

    return loss_dict


def Generator_Regularize(
    path_batch_size,
    path_regularize,
    mean_path_length,
    modelG,
    latent_dim,
    g_reg_every,
    path_batch_shrink,
    mixing=0,
):

    loss_dict = {}

    noise = mixing_noise(path_batch_size, latent_dim, mixing)

    fake_img, latents, _ = modelG(noise, return_latents=True)

    path_loss, mean_path_length, path_lengths = g_path_regularize(
        fake_img, latents, mean_path_length
    )

    weighted_path_loss = path_regularize * g_reg_every * path_loss

    # if path_batch_shrink:
    #    (weighted_path_loss + 0 * fake_img[0, 0, 0, 0]).backward()

    modelG.zero_grad(set_to_none=True)

    (weighted_path_loss + 0 * fake_img[0, 0, 0, 0]).backward()

    loss_dict["path"] = path_loss
    loss_dict["path_length"] = path_lengths.mean()

    return loss_dict, mean_path_length
