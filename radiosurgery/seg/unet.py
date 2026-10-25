"""A small 3D U-Net, sized to train on a CPU.

The MRI slices are about 1.5 mm apart but the pixels are 0.4 mm, so the
first level of the fine model uses in-plane (1 x 3 x 3) kernels and pools
only in-plane, as in the 2.5D network of the dataset paper. Deeper levels
are close to isotropic and use full 3 x 3 x 3 kernels.
"""

import torch
from torch import nn


class Block(nn.Sequential):
    def __init__(self, cin, cout, kernel):
        pad = tuple(k // 2 for k in kernel)
        super().__init__(
            nn.Conv3d(cin, cout, kernel, padding=pad, bias=False), nn.InstanceNorm3d(cout, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(cout, cout, kernel, padding=pad, bias=False), nn.InstanceNorm3d(cout, affine=True),
            nn.LeakyReLU(0.01, inplace=True))


class UNet(nn.Module):
    """One input channel (the MRI), one output channel (tumour logit).

    channels[i] features at level i; kernels[i] the convolution size there;
    pools[i] the downsampling factor from level i to i + 1.
    """

    def __init__(self, channels=(16, 32, 64, 128), kernels=((1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
                 pools=((1, 2, 2), (2, 2, 2), (2, 2, 2))):
        super().__init__()
        assert len(channels) == len(kernels) == len(pools) + 1
        self.pools = pools
        self.down = nn.ModuleList()
        cin = 1
        for c, k in zip(channels, kernels):
            self.down.append(Block(cin, c, k))
            cin = c
        self.pool = nn.ModuleList(nn.MaxPool3d(p, p) for p in pools)
        self.up = nn.ModuleList(nn.ConvTranspose3d(channels[i + 1], channels[i], p, p) for i, p in enumerate(pools))
        self.dec = nn.ModuleList(Block(2 * channels[i], channels[i], kernels[i]) for i in range(len(pools)))
        self.head = nn.Conv3d(channels[0], 1, 1)

    @property
    def divisor(self):
        """Patch sizes must be multiples of this along (z, y, x)."""
        d = [1, 1, 1]
        for p in self.pools:
            d = [a * b for a, b in zip(d, p)]
        return tuple(d)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.down):
            x = block(x)
            if i < len(self.pool):
                skips.append(x)
                x = self.pool[i](x)
        for i in reversed(range(len(self.dec))):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], dim=1))
        return self.head(x)
