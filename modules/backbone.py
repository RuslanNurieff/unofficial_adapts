from collections.abc import Sequence
from functools import partial

import hiera
import torch
import torch.nn.functional as F
from hiera.hiera_utils import Reroll, Unroll
from torch import Tensor, nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# image models of https://github.com/facebookresearch/hiera (pretrained at 224)
HIERA_MODELS = (
    "hiera_tiny_224",
    "hiera_small_224",
    "hiera_base_224",
    "hiera_base_plus_224",
    "hiera_large_224",
    "hiera_huge_224",
)


def resize_pos_embed(pos: Tensor, size: Sequence[int], unit: Sequence[int]) -> Tensor:
    """Resize a [1, C, h, w] position embedding to `size`, one mask unit at a time.

    Hiera's pretrained embedding is close to a pattern tiled once per mask unit, so plain
    interpolation shifts it out of line with the units and breaks the features (Bolya et
    al., "Window Attention is Bugged", 2023). Instead, every offset inside a unit is
    interpolated over the grid of units, which keeps the tiling intact.
    """
    _, C, h0, w0 = pos.shape
    uy, ux = unit
    p = pos.view(C, h0 // uy, uy, w0 // ux, ux).permute(2, 4, 0, 1, 3)
    p = p.reshape(uy * ux, C, h0 // uy, w0 // ux)
    p = F.interpolate(
        p, size=(size[0] // uy, size[1] // ux), mode="bicubic", align_corners=False
    )
    p = p.view(uy, ux, C, size[0] // uy, size[1] // ux).permute(2, 3, 0, 4, 1)
    return p.reshape(1, C, *size)


class HieraStem(nn.Module):
    """Patch embedding plus position embedding, as a [B, C, H/4, W/4] map."""

    def __init__(self, net: hiera.Hiera):
        super().__init__()
        self.proj = net.patch_embed.proj
        self.unit = tuple(net.mask_unit_size)
        self.multiple = tuple(p * u for p, u in zip(net.patch_stride, self.unit))
        h, w = net.tokens_spatial_shape
        self.pos_embed = nn.Parameter(
            net.pos_embed.detach().reshape(1, h, w, -1).permute(0, 3, 1, 2).contiguous()
        )

    def forward(self, x: Tensor):
        if x.shape[-2] % self.multiple[0] or x.shape[-1] % self.multiple[1]:
            raise ValueError(
                f"Hiera needs image sizes divisible by {self.multiple}, "
                f"got {tuple(x.shape[-2:])}"
            )
        x = self.proj(x)
        pos = self.pos_embed
        if pos.shape[-2:] != x.shape[-2:]:
            pos = resize_pos_embed(pos, x.shape[-2:], self.unit)
        return x + pos


class ChannelNorm(nn.Module):
    """Applies a norm over channels (e.g. LayerNorm) at every position of a map."""

    def __init__(self, norm: nn.Module):
        super().__init__()
        self.norm = norm

    def forward(self, x: Tensor):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class HieraStage(nn.Module):
    """A Hiera stage as a [B, C, H, W] -> [B, C', H', W'] map.

    Hiera runs on "unrolled" tokens, ordered so that every mask unit is contiguous and
    pooling by a stride is a max over the leading chunk (see hiera_utils.Unroll). The
    input map is unrolled by the strides still to be pooled from this stage on, and the
    output is rerolled into a map, so adapters can work on it like on a CNN feature.
    """

    def __init__(self, blocks: Sequence[nn.Module], schedule: Sequence[Sequence[int]]):
        super().__init__()
        self.blocks = nn.Sequential(*blocks)
        self.schedule = list(schedule)
        # the first block of a stage q-pools by the next stride of the schedule
        self.pool = blocks[0].attn.q_stride > 1
        self.out_schedule = self.schedule[1:] if self.pool else self.schedule

    def forward(self, x: Tensor):
        B, C, H, W = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        tokens = self.blocks(Unroll((H, W), (1, 1), self.schedule)(tokens))
        if self.pool:
            H, W = H // self.schedule[0][0], W // self.schedule[0][1]
        # Reroll keys its schedules by block; with a single entry it just undoes out_schedule
        reroll = Reroll((H, W), (1, 1), self.out_schedule, stage_ends=[0], q_pool=0)
        return reroll(tokens, 0).permute(0, 3, 1, 2)


def hiera_model(name: str):
    # default checkpoint: MAE pretrained, then fine-tuned on ImageNet-1k
    net = getattr(hiera, name)(pretrained=True)
    schedule = [net.q_stride] * (len(net.stage_ends) - 1)
    stages, start = [], 0
    for end in net.stage_ends:
        stages.append(HieraStage(net.blocks[start : end + 1], schedule))
        schedule, start = stages[-1].out_schedule, end + 1
    channels = [stage.blocks[-1].dim_out for stage in stages]
    # Hiera's residual stream has a few huge channels (stage 3: one channel holds ~half
    # of ||f||^2 and varies across positions) that would dominate the channel-normalized
    # distances. Compare stages after the LayerNorm the next block reads them with,
    # whose learned scales damp those channels.
    norms = [net.blocks[end + 1].norm1 for end in net.stage_ends[:-1]] + [net.norm]
    return HieraStem(net), stages, channels, [ChannelNorm(n) for n in norms]


BACKBONES = {
    **{name: partial(hiera_model, name) for name in HIERA_MODELS},
}


class Backbone(nn.Module):
    """Frozen backbone, run stage by stage on [B, C, H, W] maps so that the student
    can re-enter it after an adapter."""

    def __init__(self, name: str = "wide_resnet50_2", max_block: int = 3):
        super().__init__()
        if name not in BACKBONES:
            raise ValueError(f"unknown backbone {name!r}, options: {list(BACKBONES)}")
        stem, stages, channels, readouts = BACKBONES[name]()
        if not 1 <= max_block <= len(stages):
            raise ValueError(
                f"{name} has {len(stages)} blocks, got max_block={max_block}"
            )
        self.max_block = max_block
        self.channels = dict(enumerate(channels[:max_block], start=1))

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        self.stem = stem
        self.blocks = nn.ModuleList(stages[:max_block])
        # what teacher and student are compared on; the student re-enters the next
        # block with the raw adapted features
        self.readouts = nn.ModuleList(readouts[:max_block])
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    def forward(self, x: Tensor, adapters: nn.ModuleDict):
        first, last = min(adapters.as_dict()), max(adapters.as_dict())
        teacher, student = {}, {}

        with torch.no_grad():
            h = self.stem((x - self.mean) / self.std)
            for i in range(1, last + 1):
                h = self.blocks[i - 1](h)
                if i == first:
                    s = h
                if i in adapters:
                    teacher[i] = self.readouts[i - 1](h)

        for i in range(first, last + 1):
            if i > first:
                s = self.blocks[i - 1](s)
            if i in adapters:
                s = adapters[i](s)
                student[i] = self.readouts[i - 1](s)

        return teacher, student
