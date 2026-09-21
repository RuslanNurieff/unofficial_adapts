from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision import models


def ch_normalization(x: Tensor):
    return F.normalize(x, p=2, dim=1)


def feature_differences(teacher_layer: Tensor, student_layer: Tensor):
    teacher_n, student_n = (
        ch_normalization(teacher_layer),
        ch_normalization(student_layer),
    )
    diff = (teacher_n - student_n) ** 2
    return diff
