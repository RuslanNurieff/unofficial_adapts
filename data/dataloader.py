import glob
import os
from collections.abc import Iterable

import cv2
import numpy as np
import torch
from cv2.typing import MatLike
from torch.utils.data import Dataset

from .perlin import rand_perlin_2d_np

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def _rand(low: float = 0.0, high: float = 1.0) -> float:
    # torch RNG so every DataLoader worker gets its own stream
    return low + (high - low) * torch.rand(1).item()


def _gamma(img):
    return np.clip(img, 0, 1) ** _rand(0.5, 2.0)


def _brightness(img):
    return img * _rand(0.8, 1.2) + _rand(-30, 30) / 255.0


def _sharpness(img):
    blurred = cv2.GaussianBlur(img, (0, 0), 1.0)
    return img + _rand(0.0, 2.0) * (img - blurred)


def _hue_saturation(img):
    hsv = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0] + _rand(-50, 50)) % 360
    hsv[..., 1] = np.clip(hsv[..., 1] + _rand(-50, 50) / 255.0, 0, 1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _solarize(img):
    return np.where(img > 0.5, 1 - img, img)


def _posterize(img):
    levels = 2 ** torch.randint(2, 5, (1,)).item()
    return np.floor(img * levels) / levels


def _invert(img):
    return 1 - img


def _equalize(img):
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return np.dstack([cv2.equalizeHist(u8[..., c]) for c in range(3)]) / 255.0


# same family of ops as DRAEM's anomaly-source augmenters
_AUGMENTERS = (
    _gamma,
    _brightness,
    _sharpness,
    _hue_saturation,
    _solarize,
    _posterize,
    _invert,
    _equalize,
)


def _rotate(img, max_angle):
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), _rand(-max_angle, max_angle), 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


class TrainGenerator(Dataset):
    def __init__(
        self,
        img_root: str | os.PathLike,
        anomaly_source_path: str | os.PathLike,
        resize_shape: Iterable[int],
        anomaly_prob: float = 0.5,
    ):
        super().__init__()
        self.img_root = img_root
        self.image_paths = sorted(glob.glob(os.path.join(img_root, "*.png")))
        self.anomaly_source_paths = sorted(
            glob.glob(os.path.join(anomaly_source_path, "*", "*.jpg"))
        )
        if not self.image_paths:
            raise FileNotFoundError(f"no .png images in {img_root}")
        if not self.anomaly_source_paths:
            raise FileNotFoundError(f"no .jpg anomaly sources in {anomaly_source_path}")
        self.resize_shape = tuple(resize_shape)  # (height, width)
        self.anomaly_prob = anomaly_prob

    def __len__(self):
        return len(self.image_paths)

    def _read_rgb(self, path: str | os.PathLike):
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        # cv2 dsize is (width, height)
        img = cv2.resize(img, dsize=(self.resize_shape[1], self.resize_shape[0]))
        return img.astype(np.float32) / 255.0

    def _augment_source(self, img: np.ndarray):
        picks = torch.randperm(len(_AUGMENTERS))[:3].tolist()
        for i in picks:
            img = _AUGMENTERS[i](img)
        return np.clip(img, 0, 1).astype(np.float32)

    def _add_perlin(self, image: MatLike, anomaly_source_path: os.PathLike):
        if _rand() >= self.anomaly_prob:
            return (
                image,
                np.zeros((*image.shape[:2], 1), dtype=np.float32),
                np.array([0.0], dtype=np.float32),
            )

        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = self._augment_source(self._read_rgb(anomaly_source_path))

        perlin_scalex = 2 ** torch.randint(min_perlin_scale, perlin_scale, (1,)).item()
        perlin_scaley = 2 ** torch.randint(min_perlin_scale, perlin_scale, (1,)).item()

        perlin_noise = rand_perlin_2d_np(
            (self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley)
        )
        perlin_noise = _rotate(perlin_noise.astype(np.float32), 90)
        threshold = 0.5
        msk = (perlin_noise > threshold).astype(np.float32)[..., None]

        beta = _rand() * 0.8
        augmented_image = (
            image * (1 - msk) + (1 - beta) * anomaly_source_img * msk + beta * image * msk
        )

        has_anomaly = 1.0 if msk.sum() > 0 else 0.0
        return (
            augmented_image.astype(np.float32),
            msk,
            np.array([has_anomaly], dtype=np.float32),
        )

    def __getitem__(self, idx):
        rand_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        img = self._read_rgb(self.image_paths[idx])
        anomaly_source_path = self.anomaly_source_paths[rand_idx]

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
