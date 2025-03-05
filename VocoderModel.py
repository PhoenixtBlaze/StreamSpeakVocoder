import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm
from torch.nn.utils.parametrizations import weight_norm
import torch.nn.utils.parametrize as parametrize  # Ensure safe removal of weight norm


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=[1, 3, 5]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=d)
            for d in dilations
        ])
        self.gate_convs = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=d)
            for d in dilations
        ])

    def forward(self, x):
        res = x
        for conv, gate in zip(self.convs, self.gate_convs):
            out = torch.sigmoid(gate(x)) * F.leaky_relu(conv(x), 0.2)
        return out + res  # Residual connection

    def remove_weight_norm(self):
        for conv in self.convs:
            if parametrize.is_parametrized(conv):  #Check if weight_norm exists before removing
                remove_weight_norm(conv)
        for gate in self.gate_convs:
            if parametrize.is_parametrized(gate):  #Prevents crashes
                remove_weight_norm(gate)


# Generator Model
class Generator(nn.Module):
    def __init__(self, mel_channels=100, upsample_rates=[5, 5, 4, 3, 2], hidden_channels=512):
        super().__init__()
        #self.mel_channels = mel_channels
        
        self.pre_conv = (nn.Conv1d(mel_channels, hidden_channels, kernel_size=7, padding=3))
        
        # Transposed convolution upsampling
        self.upsample_blocks = nn.ModuleList()
        for rate in upsample_rates:
            self.upsample_blocks.append(nn.Sequential(
                (nn.ConvTranspose1d(hidden_channels, hidden_channels // 2, kernel_size=rate * 3, stride=rate, padding=rate // 2)),
                nn.LeakyReLU(0.2)
            ))
            hidden_channels //= 2
        
        self.resblocks = nn.ModuleList([ResidualBlock(hidden_channels) for _ in range(3)])
        
        self.post_conv = nn.Conv1d(hidden_channels, 1, kernel_size=7, padding=3)

    def forward(self, mel):
        mel = (mel - mel.min()) / (torch.clamp(mel.max() - mel.min(), min=1e-5))
        x = F.leaky_relu(self.pre_conv(mel), 0.2)
        for block in self.upsample_blocks:
            x = block(x)
        for resblock in self.resblocks:
            x = resblock(x)
        x = self.post_conv(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        if isinstance(self.pre_conv, torch.nn.Conv1d) and parametrize.is_parametrized(self.pre_conv):
            remove_weight_norm(self.pre_conv)

        for block in self.upsample_blocks:
            if isinstance(block, torch.nn.ConvTranspose1d) and parametrize.is_parametrized(block):
                remove_weight_norm(block)

        for resblock in self.resblocks:
            resblock.remove_weight_norm()


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_conv = weight_norm(nn.Conv1d(1, 64, kernel_size=15, stride=1, padding=7))
        self.layers = nn.ModuleList([
            weight_norm(nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)),
            weight_norm(nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2)),
            weight_norm(nn.Conv1d(256, 512, kernel_size=5, stride=2, padding=2)),
            weight_norm(nn.Conv1d(512, 1024, kernel_size=5, stride=2, padding=2)),
        ])
        self.final_conv = weight_norm(nn.Conv1d(1024, 1, kernel_size=5, stride=1, padding=2))

    def forward(self, x, return_intermediate=False):
        x = F.leaky_relu(self.input_conv(x), 0.2)
        features = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), 0.2)
            if isinstance(return_intermediate, bool) and return_intermediate:
                features.append(x)
        x = self.final_conv(x)
        return (x, features) if return_intermediate is True else x  # Ensure it's a boolean

    def remove_weight_norm(self):
        remove_weight_norm(self.input_conv)
        for layer in self.layers:
            remove_weight_norm(layer)
        remove_weight_norm(self.final_conv)

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            Discriminator(),
            Discriminator(),
            Discriminator()
        ])
        self.pools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, x, return_intermediate=False):
        y_outputs = []
        fmap_outputs = []
        for i, d in enumerate(self.discriminators):
            if i > 0:
                x = self.pools[i - 1](x)
            y, fmap = d(x, return_intermediate=return_intermediate)  # Unpack discriminator output
            y_outputs.append(y)
            fmap_outputs.append(fmap)
        return y_outputs, fmap_outputs  # Ensure it returns exactly 4 values



class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(kernel_size // 2, 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # Reshape 1D audio to 2D with `period` segments
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        
        x = x.view(b, c, t // self.period, self.period)  # [B, 1, T//P, P]

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, 0.2)
            fmap.append(x)
        
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap



class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
