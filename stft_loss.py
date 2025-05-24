# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Original copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

import torch
import torch.nn.functional as F


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=False)
    return x_stft

def magnitude(x_stft):
    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)

class ComplexSpectralLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_stft, y_stft):
        x_real, y_real = x_stft[..., 0], y_stft[..., 0]
        x_imag, y_imag = x_stft[..., 1], y_stft[..., 1]
        _loss = F.l1_loss(torch.sign(y_real) * torch.log(1.+torch.abs(y_real)), torch.sign(x_real) * torch.log(1.+torch.abs(x_real)))
        _loss += F.l1_loss(torch.sign(y_imag) * torch.log(1.+torch.abs(y_imag)), torch.sign(x_imag) * torch.log(1.+torch.abs(x_imag)))
        return _loss

class ComplexSpectralConvergenceLoss(torch.nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x_stft: torch.Tensor, y_stft: torch.Tensor) -> torch.Tensor:
        X_mag = torch.abs(x_stft)  # [B, F, T]
        Y_mag = torch.abs(y_stft)

        sc_num = torch.norm(Y_mag - X_mag, p='fro')
        sc_den = torch.norm(Y_mag,      p='fro').clamp_min(self.eps)
        sc_loss = sc_num / sc_den

        log_X = torch.log1p(X_mag)
        log_Y = torch.log1p(Y_mag)
        mag_loss = F.l1_loss(log_X, log_Y)

        return sc_loss + mag_loss
        
class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        #self.complex_loss = ComplexSpectralLoss()
        self.complex_loss = ComplexSpectralConvergenceLoss()
        
    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_stft = stft(x.cuda(), self.fft_size, self.shift_size, self.win_length, self.window)
        y_stft = stft(y.cuda(), self.fft_size, self.shift_size, self.win_length, self.window)
        x_mag = magnitude(x_stft)
        y_mag = magnitude(y_stft)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
        comp_loss = self.complex_loss(x_stft, y_stft)
        return sc_loss, mag_loss, comp_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window", factor_sc=0.1, factor_mag=0.1, factor_cmp=0.1):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            factor (float): a balancing factor across different losses.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag
        self.factor_cmp = factor_cmp
        
    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        cmp_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l, cmp_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
            cmp_loss += cmp_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)
        cmp_loss /= len(self.stft_losses)

        return self.factor_sc*sc_loss, self.factor_mag*mag_loss, self.factor_cmp*cmp_loss
