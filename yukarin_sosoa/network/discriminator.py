"""Discriminatorネットワークモジュール（HiFi-GANベース）"""

import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import AvgPool1d, Conv1d, Conv2d
from torch.nn.utils import spectral_norm, weight_norm

from .vocoder import get_padding

LRELU_SLOPE = 0.1


class DiscriminatorP(nn.Module):
    """Period-based Discriminator"""

    def __init__(
        self,
        period: int,
        initial_channel: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        channels = [
            initial_channel,
            initial_channel * 4,
            initial_channel * 16,
            initial_channel * 32,
            initial_channel * 32,
        ]
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        channels[0],
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        channels[0],
                        channels[1],
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        channels[1],
                        channels[2],
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        channels[2],
                        channels[3],
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        channels[3], channels[4], (kernel_size, 1), 1, padding=(2, 0)
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(channels[4], 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:  # noqa: D102
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = x.flatten(1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """複数のPeriod-based Discriminatorを組み合わせたもの"""

    def __init__(self, initial_channel: int):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(period=2, initial_channel=initial_channel),
                DiscriminatorP(period=3, initial_channel=initial_channel),
                DiscriminatorP(period=5, initial_channel=initial_channel),
                DiscriminatorP(period=7, initial_channel=initial_channel),
                DiscriminatorP(period=11, initial_channel=initial_channel),
            ]
        )

    def forward(  # noqa: D102
        self,
        y: Tensor,  # (B, 1, wL)
        y_hat: Tensor,  # (B, 1, wL)
    ) -> tuple[list[Tensor], list[Tensor], list[list[Tensor]], list[list[Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for discriminator in self.discriminators:
            y_d_r, fmap_r = discriminator(y)
            y_d_g, fmap_g = discriminator(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(nn.Module):
    """Scale-based Discriminator"""

    def __init__(
        self,
        initial_channel: int,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        channels = [
            initial_channel,
            initial_channel,
            initial_channel * 2,
            initial_channel * 4,
            initial_channel * 8,
            initial_channel * 8,
            initial_channel * 8,
        ]
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, channels[0], 15, 1, padding=7)),
                norm_f(Conv1d(channels[0], channels[1], 41, 2, groups=4, padding=20)),
                norm_f(Conv1d(channels[1], channels[2], 41, 2, groups=16, padding=20)),
                norm_f(Conv1d(channels[2], channels[3], 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(channels[3], channels[4], 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(channels[4], channels[5], 41, 1, groups=16, padding=20)),
                norm_f(Conv1d(channels[5], channels[6], 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(channels[6], 1, 3, 1, padding=1))

    def forward(  # noqa: D102
        self,
        x: Tensor,  # (B, 1, wL)
    ) -> tuple[Tensor, list[Tensor]]:
        fmap = []
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = x.flatten(1, -1)

        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    """複数のScale-based Discriminatorを組み合わせたもの"""

    def __init__(self, initial_channel: int):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(initial_channel=initial_channel, use_spectral_norm=True),
                DiscriminatorS(initial_channel=initial_channel),
                DiscriminatorS(initial_channel=initial_channel),
            ]
        )
        self.meanpools = nn.ModuleList(
            [AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)]
        )

    def forward(  # noqa: D102
        self,
        y: Tensor,  # (B, 1, wL)
        y_hat: Tensor,  # (B, 1, wL)
    ) -> tuple[list[Tensor], list[Tensor], list[list[Tensor]], list[list[Tensor]]]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, discriminator in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = discriminator(y)
            y_d_g, fmap_g = discriminator(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
