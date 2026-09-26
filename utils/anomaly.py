from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def ch_normalization(x: Tensor):
    return F.normalize(x, p=2, dim=1)


def feature_differences(teacher_layer: Tensor, student_layer: Tensor):
    teacher_n, student_n = (
        ch_normalization(teacher_layer),
        ch_normalization(student_layer),
    )
    diff = (teacher_n - student_n) ** 2
    return diff


def resize_to(x: Tensor, size: Sequence[int]):
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    return F.interpolate(x, tuple(size), mode="bilinear", align_corners=False)


def anomaly_map(
    teacher: Mapping[int, Tensor],
    student: Mapping[int, Tensor],
    size: Sequence[int],
):
    # inference map (paper Fig. 2c): per-block diff summed over channels,
    # upsampled to the input size and summed over blocks
    amap = 0
    for block in teacher:
        d = feature_differences(teacher[block], student[block])
        amap = amap + resize_to(d.sum(dim=1, keepdim=True), size)
    return amap


def gaussian_blur(x: Tensor, sigma: float = 4.0):
    radius = int(4 * sigma + 0.5)
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    # separable blur, reflect padding keeps borders from darkening
    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, kernel.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(x, kernel.view(1, 1, -1, 1))
