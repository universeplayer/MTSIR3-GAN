"""
R3GAN-1D: 1D adaptation of R3GAN for time series imputation refinement.

Adapts the R3GAN architecture (NeurIPS 2024) from 2D image generation to 1D
time series refinement. Key changes from original:
- Conv2d → Conv1d (time series is 1D spatial)
- 2D up/downsample → 1D interpolation-based resampling
- Reference implementations (no CUDA custom ops dependency)
- Conditioned on coarse imputation: Generator learns residual refinement
- Added frequency-domain discriminator branch

Architecture follows the original R3GAN:
- ResNet-style blocks with Fixup initialization (no normalization layers)
- Grouped convolutions with inverted bottlenecks
- Interpolative resampling with lowpass filtering
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# Building blocks
# ============================================================

def msr_init(layer, activation_gain=1.0):
    """Modified variance scaling initialization (MSR/Fixup-style)."""
    fan_in = layer.weight.data.size(1) * layer.weight.data[0][0].numel()
    layer.weight.data.normal_(0, activation_gain / math.sqrt(fan_in))
    if layer.bias is not None:
        layer.bias.data.zero_()
    return layer


class BiasedActivation(nn.Module):
    """Bias + LeakyReLU fused activation (reference implementation)."""
    GAIN = math.sqrt(2 / (1 + 0.2 ** 2))

    def __init__(self, channels):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        # x: (B, C, T)
        return self.act(x + self.bias.view(1, -1, 1))


class Conv1d(nn.Module):
    """1D convolution with MSR initialization."""

    def __init__(self, in_ch, out_ch, kernel_size, groups=1, activation_gain=1.0):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.layer = msr_init(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, groups=groups, bias=False),
            activation_gain=activation_gain,
        )

    def forward(self, x):
        return F.conv1d(x, self.layer.weight.to(x.dtype),
                        padding=self.layer.padding, groups=self.layer.groups)


class ResidualBlock1D(nn.Module):
    """
    Inverted-bottleneck residual block (1D).
    Structure: 1x1 expand → KxK grouped conv → 1x1 project, with Fixup scaling.
    """

    def __init__(self, channels, cardinality, expansion, kernel_size, var_scale):
        super().__init__()
        n_linear = 3
        expanded = channels * expansion
        gain = BiasedActivation.GAIN * var_scale ** (-1 / (2 * n_linear - 2))

        self.conv1 = Conv1d(channels, expanded, 1, activation_gain=gain)
        self.conv2 = Conv1d(expanded, expanded, kernel_size, groups=cardinality, activation_gain=gain)
        self.conv3 = Conv1d(expanded, channels, 1, activation_gain=0)
        self.act1 = BiasedActivation(expanded)
        self.act2 = BiasedActivation(expanded)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(self.act1(y))
        y = self.conv3(self.act2(y))
        return x + y


# ============================================================
# 1D Resampling (lowpass-filtered, anti-aliased)
# ============================================================

def _create_lowpass_1d(weights):
    """Create 1D lowpass kernel from filter weights."""
    k = np.convolve(weights, [1, 1])
    k = torch.tensor(k, dtype=torch.float32)
    return k / k.sum()


class Upsample1D(nn.Module):
    """2x upsample with lowpass filtering (anti-aliased)."""

    def __init__(self, in_ch, out_ch, filter_weights=(1, 2, 1)):
        super().__init__()
        kernel = _create_lowpass_1d(filter_weights)
        self.register_buffer('kernel', kernel)
        if in_ch != out_ch:
            self.proj = Conv1d(in_ch, out_ch, 1)
        else:
            self.proj = None

    def forward(self, x):
        if self.proj is not None:
            x = self.proj(x)
        # Nearest-neighbor 2x upsample then lowpass filter
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # Apply 1D lowpass filter per channel
        k = self.kernel.view(1, 1, -1).to(x.dtype)
        pad = (k.shape[-1] - 1) // 2
        B, C, T = x.shape
        x = F.conv1d(x.reshape(B * C, 1, T), k, padding=pad).reshape(B, C, -1)
        return x


class Downsample1D(nn.Module):
    """2x downsample with lowpass filtering (anti-aliased)."""

    def __init__(self, in_ch, out_ch, filter_weights=(1, 2, 1)):
        super().__init__()
        kernel = _create_lowpass_1d(filter_weights)
        self.register_buffer('kernel', kernel)
        if in_ch != out_ch:
            self.proj = Conv1d(in_ch, out_ch, 1)
        else:
            self.proj = None

    def forward(self, x):
        # Lowpass filter then 2x downsample
        k = self.kernel.view(1, 1, -1).to(x.dtype)
        pad = (k.shape[-1] - 1) // 2
        B, C, T = x.shape
        x = F.conv1d(x.reshape(B * C, 1, T), k, padding=pad).reshape(B, C, -1)
        x = x[:, :, ::2]  # stride-2 downsample
        if self.proj is not None:
            x = self.proj(x)
        return x


# ============================================================
# Generator
# ============================================================

class GeneratorStage(nn.Module):
    def __init__(self, in_ch, out_ch, cardinality, n_blocks, expansion,
                 kernel_size, var_scale, upsample=False):
        super().__init__()
        layers = []
        if upsample:
            layers.append(Upsample1D(in_ch, out_ch))
        elif in_ch != out_ch:
            layers.append(Conv1d(in_ch, out_ch, 1))
        for _ in range(n_blocks):
            layers.append(ResidualBlock1D(out_ch, cardinality, expansion, kernel_size, var_scale))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class RefinerGenerator(nn.Module):
    """
    R3GAN-1D Generator adapted for imputation refinement.

    Input: [X_observed, mask, X_coarse, noise] concatenated along channel dim
           Shape: (B, 3*F + noise_dim, T)
    Output: residual (B, F, T), then final = X_coarse + residual
    """

    def __init__(self, n_features, noise_dim=32,
                 width=(256, 256, 256, 256),
                 blocks=(2, 2, 2, 2),
                 cardinality=(64, 64, 64, 64),
                 expansion=2, kernel_size=3):
        super().__init__()
        self.n_features = n_features
        self.noise_dim = noise_dim
        in_channels = 3 * n_features + noise_dim
        var_scale = sum(blocks)

        # Encoder path (downsample)
        self.encoder = nn.ModuleList()
        ch = in_channels
        encoder_channels = []
        for i in range(len(width)):
            self.encoder.append(GeneratorStage(
                ch, width[i], cardinality[i], blocks[i], expansion,
                kernel_size, var_scale, upsample=False,
            ))
            encoder_channels.append(width[i])
            ch = width[i]

        # Decoder path (same resolution, no upsampling — we're a refiner, not a generator)
        # Just additional residual blocks for refinement
        self.decoder = nn.ModuleList()
        for i in reversed(range(len(width))):
            # Skip connection from encoder doubles channels
            self.decoder.append(GeneratorStage(
                width[i] + encoder_channels[i], width[i],
                cardinality[i], blocks[i], expansion,
                kernel_size, var_scale, upsample=False,
            ))

        # Output projection: residual
        self.out_proj = Conv1d(width[0], n_features, 1, activation_gain=0)

    def forward(self, X_observed, mask, X_coarse, noise=None):
        """
        Args:
            X_observed: (B, T, F)
            mask: (B, T, F)
            X_coarse: (B, T, F)
            noise: (B, T, noise_dim) or None

        Returns:
            X_refined: (B, T, F) - final imputed values
        """
        B, T, F = X_observed.shape

        if noise is None:
            noise = torch.randn(B, T, self.noise_dim, device=X_observed.device)

        # Concatenate inputs: (B, T, 3F + noise_dim) → (B, C, T) for Conv1d
        x = torch.cat([X_observed, mask, X_coarse, noise], dim=-1)
        x = x.permute(0, 2, 1)  # (B, C, T)

        # Encoder
        enc_features = []
        for stage in self.encoder:
            x = stage(x)
            enc_features.append(x)

        # Decoder with skip connections
        for i, stage in enumerate(self.decoder):
            skip = enc_features[len(enc_features) - 1 - i]
            x = torch.cat([x, skip], dim=1)
            x = stage(x)

        # Output residual
        residual = self.out_proj(x).permute(0, 2, 1)  # (B, T, F)

        # Residual learning: refined = coarse + learned_residual
        X_refined = X_coarse + residual

        # Preserve observed values
        X_final = X_observed * mask + X_refined * (1 - mask)
        return X_final


# ============================================================
# Discriminator
# ============================================================

class DiscriminatorStage(nn.Module):
    def __init__(self, in_ch, out_ch, cardinality, n_blocks, expansion,
                 kernel_size, var_scale, downsample=False):
        super().__init__()
        layers = []
        for _ in range(n_blocks):
            layers.append(ResidualBlock1D(in_ch, cardinality, expansion, kernel_size, var_scale))
        if downsample:
            layers.append(Downsample1D(in_ch, out_ch))
        elif in_ch != out_ch:
            layers.append(Conv1d(in_ch, out_ch, 1))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class RefinerDiscriminator(nn.Module):
    """
    R3GAN-1D Discriminator with optional frequency-domain branch.

    Input: complete time series (B, T, F)
    Output: scalar realness logit per sample
    """

    def __init__(self, n_features,
                 width=(256, 256, 256, 256),
                 blocks=(2, 2, 2, 2),
                 cardinality=(64, 64, 64, 64),
                 expansion=2, kernel_size=3,
                 freq_branch=True):
        super().__init__()
        var_scale = sum(blocks)
        self.freq_branch = freq_branch

        # Time-domain path
        self.extraction = Conv1d(n_features, width[0], 1)
        self.stages = nn.ModuleList()
        for i in range(len(width) - 1):
            self.stages.append(DiscriminatorStage(
                width[i], width[i + 1], cardinality[i], blocks[i], expansion,
                kernel_size, var_scale, downsample=True,
            ))
        # Final stage (no downsample, global pool)
        self.stages.append(DiscriminatorStage(
            width[-1], width[-1], cardinality[-1], blocks[-1], expansion,
            kernel_size, var_scale, downsample=False,
        ))
        self.pool = nn.AdaptiveAvgPool1d(1)
        time_out_dim = width[-1]

        # Frequency-domain branch
        if freq_branch:
            self.freq_extraction = Conv1d(n_features, width[0] // 2, 1)
            self.freq_stages = nn.ModuleList([
                DiscriminatorStage(
                    width[0] // 2, width[0] // 2, cardinality[0] // 2,
                    1, expansion, kernel_size, var_scale, downsample=True,
                ),
                DiscriminatorStage(
                    width[0] // 2, width[0] // 2, cardinality[0] // 2,
                    1, expansion, kernel_size, var_scale, downsample=False,
                ),
            ])
            self.freq_pool = nn.AdaptiveAvgPool1d(1)
            freq_out_dim = width[0] // 2
        else:
            freq_out_dim = 0

        self.head = nn.Linear(time_out_dim + freq_out_dim, 1, bias=False)

    def forward(self, x, return_features=False):
        """
        Args:
            x: (B, T, F) - complete time series (real or fake)
        Returns:
            logits: (B,) - realness logits
        """
        x_t = x.permute(0, 2, 1)  # (B, F, T)

        # Time-domain path
        h = self.extraction(x_t)
        for stage in self.stages:
            h = stage(h)
        h_time = self.pool(h).squeeze(-1)  # (B, C)

        if self.freq_branch:
            # Frequency-domain path: FFT magnitude spectrum
            x_freq = torch.fft.rfft(x_t, dim=-1).abs()  # (B, F, T//2+1)
            h_f = self.freq_extraction(x_freq)
            for stage in self.freq_stages:
                h_f = stage(h_f)
            h_freq = self.freq_pool(h_f).squeeze(-1)  # (B, C)
            h_combined = torch.cat([h_time, h_freq], dim=-1)
        else:
            h_combined = h_time

        logits = self.head(h_combined).squeeze(-1)  # (B,)
        return logits


# ============================================================
# R3GAN Training Logic (RpGAN + R1 + R2)
# ============================================================

class R3GANTrainer:
    """
    R3GAN adversarial or discriminator-disabled reconstruction training with:
    - Relativistic paired GAN loss (RpGAN)
    - R1 gradient penalty (on real samples)
    - R2 gradient penalty (on fake samples)
    - Reconstruction loss (L1 on artificially missing positions)
    - Frequency-domain loss (L1 on FFT magnitude)

    With ``adversarial=False``, the trainer drops the discriminator entirely
    and optimizes only the masked L1 and configured spectral reconstruction
    terms. Set ``lambda_freq=0`` for a literal L1-only control.
    """

    def __init__(self, generator, discriminator,
                 lambda_recon=10.0, lambda_freq=1.0,
                 adversarial=True):
        if adversarial and discriminator is None:
            raise ValueError('adversarial training requires a discriminator')
        self.G = generator
        # In reconstruction-only mode, deliberately drop the discriminator
        # reference.  This makes the control structurally discriminator-free,
        # rather than approximating it with a large reconstruction weight.
        self.D = discriminator if adversarial else None
        self.lambda_recon = lambda_recon
        self.lambda_freq = lambda_freq
        self.adversarial = adversarial

    @staticmethod
    def gradient_penalty(samples, critics):
        """Zero-centered gradient penalty."""
        grad, = torch.autograd.grad(
            outputs=critics.sum(), inputs=samples, create_graph=True,
        )
        return grad.square().sum(dim=list(range(1, grad.ndim)))

    def generator_step(self, X_obs, mask, X_coarse, X_real,
                       gamma_recon=1.0, noise=None):
        """
        Compute generator loss and return metrics.
        Call .backward() on the returned loss externally.
        """
        if noise is None:
            # Preserve compatibility with existing generators and callers.
            X_fake = self.G(X_obs, mask, X_coarse)
        else:
            X_fake = self.G(X_obs, mask, X_coarse, noise=noise)

        if self.adversarial:
            # Adversarial: relativistic paired
            d_fake = self.D(X_fake)
            d_real = self.D(X_real.detach())
            adv_loss = F.softplus(-(d_fake - d_real)).mean()
        else:
            # Keep the metric schema stable without building or evaluating D.
            adv_loss = X_fake.new_zeros(())

        # Reconstruction on MISSING positions (the actual imputation task)
        missing_mask = 1 - mask
        n_missing = missing_mask.sum().clamp(min=1)
        recon_loss = (torch.abs(X_fake - X_real) * missing_mask).sum() / n_missing

        # Frequency-domain loss on the full sequence (real vs fake should match spectrally)
        fake_fft = torch.fft.rfft(X_fake, dim=1).abs()
        real_fft = torch.fft.rfft(X_real, dim=1).abs()
        freq_loss = F.l1_loss(fake_fft, real_fft)

        total = adv_loss + self.lambda_recon * recon_loss + self.lambda_freq * freq_loss

        return total, {
            'g_adv': adv_loss.item(),
            'g_recon': recon_loss.item(),
            'g_freq': freq_loss.item(),
            'g_total': total.item(),
        }

    def discriminator_step(self, X_obs, mask, X_coarse, X_real, gamma=0.05,
                           noise=None):
        """
        Compute discriminator loss with R1+R2 penalties.
        """
        if not self.adversarial:
            raise RuntimeError(
                'discriminator_step is unavailable in reconstruction-only mode'
            )

        X_real_gp = X_real.detach().requires_grad_(True)
        if noise is None:
            # The implicit-noise path remains available for legacy callers.
            X_fake = self.G(X_obs, mask, X_coarse)
        else:
            X_fake = self.G(X_obs, mask, X_coarse, noise=noise)
        X_fake = X_fake.detach().requires_grad_(True)

        d_real = self.D(X_real_gp)
        d_fake = self.D(X_fake)

        # Relativistic adversarial loss
        adv_loss = F.softplus(-(d_real - d_fake)).mean()

        # R1 + R2 gradient penalties
        r1 = self.gradient_penalty(X_real_gp, d_real)
        r2 = self.gradient_penalty(X_fake, d_fake)

        total = adv_loss + (gamma / 2) * (r1 + r2).mean()

        return total, {
            'd_adv': adv_loss.item(),
            'd_r1': r1.mean().item(),
            'd_r2': r2.mean().item(),
            'd_total': total.item(),
        }
