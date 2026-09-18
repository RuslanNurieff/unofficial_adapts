import glob
import os
from collections.abc import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
from cv2.typing import MatLike
from torch.utils.data import Dataset

from .perlin import rand_perlin_2d_np

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class TrainGenerator(Dataset):
    def __init__(
        self,
        img_root: str | os.PathLike,
        anomaly_source_path: str | os.PathLike,
        resize_shape: Iterable[int],
    ):
        super().__init__()
        self.img_root = img_root
        self.image_paths = sorted(glob.glob(img_root + "/*.png"))
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path + "/*/*.jpg"))
        self.resize_shape = resize_shape

    def __len__(self):
        return len(self.image_paths)

    def _add_perlin(self, image: MatLike, anomaly_source_path: os.PathLike):
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(
            anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0])
        )

        perlin_scalex = (
            2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        )
        perlin_scaley = (
            2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        )

        perlin_noise = rand_perlin_2d_np(
            (self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley)
        )
        threshold = 0.5
        perlin_thr = np.where(
            perlin_noise > threshold,
            np.ones_like(perlin_noise),
            np.zeros_like(perlin_noise),
        )
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_source_img.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = (
            image * (1 - perlin_thr)
            + (1 - beta) * img_thr
            + beta * image * (perlin_thr)
        )

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return (
                image,
                np.zeros_like(perlin_thr, dtype=np.float32),
                np.array([0.0], dtype=np.float32),
            )
        else:
            augmented_image = augmented_image.astype(np.float32)
            msk = (perlin_thr).astype(np.float32)
            augmented_image = msk * augmented_image + (1 - msk) * image
            has_anomaly = 1.0
            if np.sum(msk) == 0:
                has_anomaly = 0.0
            return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)

    def __getitem__(self, idx):
        rand_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        img = self.image_paths[idx]
        anomaly_source_path = self.anomaly_source_paths[rand_idx]

        img = cv2.imread(img)
        img = cv2.resize(img, dsize=(self.resize_shape[0], self.resize_shape[1]))
        img = img.astype(np.float32) / 255.0

        perlin_img, mask, label = self._add_perlin(img, anomaly_source_path)

        img = torch.from_numpy(img).permute(2, 0, 1)
        perlin_img = torch.from_numpy(perlin_img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).permute(2, 0, 1)

        sample = {
            "image": img,
            "anomaly_image": perlin_img,
            "anomaly_mask": mask,
            "has_anomaly": label,
        }

        return sample
