import numpy as np
import scipy
import torch
import torch.nn as nn
from torch import Tensor, from_numpy

from gan.data.statsMasked import normalizeUnderMask


class NormalizeUnderMasktorch(nn.Module):
    def __init__(
        self, mean, std, MaskV=9999.0, fill=0.0, scale=1.0, inplace=False
    ) -> None:
        super().__init__()
        self.mean = np.array(mean).reshape((len(mean), 1, 1))
        self.std = np.array(std).reshape((len(std), 1, 1))
        self.maskv = MaskV
        self.fill = fill
        self.scale = scale

    def forward(self, tensor: Tensor) -> Tensor:
        tensor = normalizeUnderMask(
            data=tensor.numpy(),
            MaskV=self.maskv,
            Mean=self.mean,
            std=self.std,
            scale=self.scale,
            fill=self.fill,
        )
        return (Tensor.transpose((1, 2, 0))).float()


class MultiOptionNormalize(object):
    def __init__(self, means, stds, maxs, mins, config, dataset_handler_yaml):
        self.means = means
        self.stds = stds
        self.maxs = maxs
        self.mins = mins
        self.config = config
        self.dataset_handler_yaml = dataset_handler_yaml
        self.gaussian_std = self.dataset_handler_yaml["rr_transform"]["gaussian_std"]
        if self.gaussian_std:
            for _ in range(
                self.dataset_handler_yaml["rr_transform"]["log_transform_iteration"]
            ):
                self.gaussian_std = np.log(1 + self.gaussian_std)
            self.gaussian_std_map = (
                np.random.choice([-1, 1], size=self.config.crop_size)
                * self.gaussian_std
            )
            self.gaussian_noise = np.mod(
                np.random.normal(0, self.gaussian_std, size=self.config.crop_size),
                self.gaussian_std_map,
            )
        if self.dataset_handler_yaml["normalization"]["type"] == "mean":
            if np.ndim(self.stds) > 1:
                if (
                    self.dataset_handler_yaml["normalization"]["for_rr"][
                        "blur_iteration"
                    ]
                    > 0
                ):
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
                    for _ in range(
                        self.dataset_handler_yaml["normalization"]["for_rr"][
                            "blur_iteration"
                        ]
                    ):
                        self.stds[0] = scipy.ndimage.convolve(
                            self.stds[0], gaussian_filter, mode="mirror"
                        )
                self.means = from_numpy(self.means)
                self.stds = from_numpy(self.stds)
            else:
                self.means = from_numpy(self.means).view(-1, 1, 1)
                self.stds = from_numpy(self.stds).view(-1, 1, 1)
        elif self.dataset_handler_yaml["normalization"]["type"] == "minmax":
            if np.ndim(self.maxs) > 1:
                if (
                    self.dataset_handler_yaml["normalization"]["for_rr"][
                        "blur_iteration"
                    ]
                    > 0
                ):
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
                    for _ in range(
                        self.dataset_handler_yaml["normalization"]["for_rr"][
                            "blur_iteration"
                        ]
                    ):
                        self.maxs[0] = scipy.ndimage.convolve(
                            self.maxs[0], gaussian_filter, mode="mirror"
                        )
                self.maxs = from_numpy(self.maxs)
                self.mins = from_numpy(self.mins)
            else:
                self.maxs = from_numpy(self.maxs).view(-1, 1, 1)
                self.mins = from_numpy(self.mins).view(-1, 1, 1)
        elif self.dataset_handler_yaml["normalization"]["type"] == "quant":
            self.maxs = from_numpy(self.maxs)
            self.mins = from_numpy(self.mins)

    def __call__(self, sample):
        if not isinstance(sample, Tensor):
            raise TypeError(
                f"Input sample should be a torch tensor. Got {type(sample)}."
            )
        if sample.ndim < 3:
            raise ValueError(
                f"Expected sample to be a tensor image of size (..., C, H, W). Got tensor.size() = {sample.size()}."
            )
        if self.gaussian_std != 0:
            mask_no_rr = sample[0].numpy() <= self.gaussian_std
            sample[0] = sample[0].add_(from_numpy(self.gaussian_noise * mask_no_rr))
        if self.dataset_handler_yaml["normalization"]["type"] == "mean":
            sample = (sample - self.means) / self.stds
        elif (
            self.dataset_handler_yaml["normalization"]["type"] == "minmax"
            or self.dataset_handler_yaml["normalization"]["type"] == "quant"
        ):
            sample = -1 + 2 * ((sample - self.mins) / (self.maxs - self.mins))
        return sample
