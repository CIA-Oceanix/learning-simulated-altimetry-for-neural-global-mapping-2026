import numpy as np
import torch


class UNet(torch.nn.Module):
    def __init__(
        self,
        dim_in,
        channel_dims,
        max_depth=None,
        dim_out=None,
        interp_mode="bilinear",
        dropout=0.1,
        activation_layer=None,
        bias=True,
    ):
        super().__init__()

        dim_out = dim_out or dim_in
        activation_layer = activation_layer or torch.nn.ReLU()

        if max_depth:
            self.max_depth = max(max_depth, len(channel_dims) // 3)
        else:
            self.max_depth = len(channel_dims) // 3

        self.ups = torch.nn.ModuleList()
        self.up_pools = torch.nn.ModuleList()
        self.downs = torch.nn.ModuleList()
        self.down_pools = torch.nn.ModuleList()
        self.residues = list()

        self.interp_mode = interp_mode
        self.dropout = dropout

        self.bottom_transform = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channel_dims[self.max_depth * 3 - 1],
                out_channels=channel_dims[self.max_depth * 3],
                padding="same",
                kernel_size=3,
                bias=bias,
            ),
            activation_layer,
            torch.nn.Dropout(p=dropout),
            torch.nn.Conv2d(
                in_channels=channel_dims[self.max_depth * 3],
                out_channels=channel_dims[self.max_depth * 3],
                padding="same",
                kernel_size=3,
                bias=bias,
            ),
            activation_layer,
        )

        for depth in range(self.max_depth):
            self.ups.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3 + 2] * 2,
                        out_channels=channel_dims[depth * 3 + 1],
                        padding="same",
                        kernel_size=3,
                        bias=bias,
                    ),
                    activation_layer,
                    torch.nn.Dropout(p=dropout),
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3 + 1],
                        out_channels=channel_dims[depth * 3],
                        padding="same",
                        kernel_size=3,
                        bias=bias,
                    ),
                    activation_layer,
                )
            )
            self.up_pools.append(
                InterpolationBasedUpsampling(
                    channels=channel_dims[depth * 3 + 3],
                    use_conv=True,
                    out_channels=channel_dims[depth * 3 + 2],
                    interp_mode=self.interp_mode,
                )
            )
            self.downs.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=dim_in
                        if depth == 0
                        else channel_dims[depth * 3 - 1],
                        out_channels=channel_dims[depth * 3],
                        padding="same",
                        kernel_size=3,
                        bias=bias,
                    ),
                    activation_layer,
                    torch.nn.Dropout(p=dropout),
                    torch.nn.Conv2d(
                        in_channels=channel_dims[depth * 3],
                        out_channels=channel_dims[depth * 3 + 1],
                        padding="same",
                        kernel_size=3,
                        bias=bias,
                    ),
                    activation_layer,
                )
            )

            self.down_pools.append(torch.nn.AvgPool2d(kernel_size=2))

        self.final_up = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=channel_dims[0],
                out_channels=4 * dim_out,
                padding="same",
                kernel_size=3,
                bias=bias,
            )
        )
        self.final_linear = torch.nn.Sequential(
            torch.nn.Linear(4 * dim_out, dim_out, bias=bias)
        )

    def forward(self, batch):
        x = batch.input.nan_to_num()

        x = self.unet_step(x, depth=0)
        x = self.final_up(x)
        x = torch.permute(x, dims=(0, 2, 3, 1))
        x = self.final_linear(x)
        x = torch.permute(x, dims=(0, 3, 1, 2))
        return x

    def unet_step(self, x, depth):
        x, residue = self.down(x, depth)
        self.residues.append(residue)

        if depth == self.max_depth - 1:
            x = self.bottom_transform(x)
        else:
            x = self.unet_step(x, depth + 1)

        return self.up(x, depth)

    def down(self, x, depth):
        x = self.downs[depth](x)
        return self.down_pools[depth](x), x

    def up(self, x, depth):
        x = self.up_pools[depth](x)
        x = self.concat_residue(x)
        return self.ups[depth](x)

    def concat_residue(self, x):
        if len(self.residues) != 0:
            residue = self.residues.pop(-1)

            _, _, h_x, w_x = x.shape
            _, _, h_r, w_r = residue.shape

            pad_h = h_r - h_x
            pad_w = w_r - w_x

            if pad_h > 0 or pad_w > 0:
                x = torch.nn.functional.pad(
                    x, (0, pad_w, 0, pad_h), mode="reflect", value=0
                )

            return torch.concat((x, residue), dim=1)
        else:
            return x


class InterpolationBasedUpsampling(torch.nn.Module):
    """
    An upsampling layer, with optional convolution output.
    """

    def __init__(
        self, channels, use_conv, out_channels=None, interp_mode="bilinear",
        bias=True, scale_factor=2,
    ):
        """
        PARAMETERS
        ----------
        channels (int):
            Number of channels in the inputs.

        use_conv (bool):
            If a convolution must be applied to the output.

        out_channels (int):
            Number of channels in the outputs.

        interp_mode (str):
            Interpolation mode to be used (bilinear by default).

        bias (bool):
            Whether the applied convolution must contain a bias term or not.

        scale_factor (int):
            Scale to be used for the upsampling.
        """
        super().__init__()

        out_channels = out_channels or channels

        self.interp_mode = interp_mode
        self.scale_factor = scale_factor

        if use_conv:
            self.conv = torch.nn.Conv2d(
                in_channels=channels,
                out_channels=out_channels,
                padding="same",
                kernel_size=1,
                bias=bias,
            )
        else:
            self.conv = torch.nn.Identity()

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=self.scale_factor, mode=self.interp_mode,
        )
        x = self.conv(x)
        return x
