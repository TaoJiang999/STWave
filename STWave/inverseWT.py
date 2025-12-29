import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
from typing import List
import warnings
warnings.filterwarnings('ignore')


class InverseDWT1D(nn.Module):
    """
        A one-dimensional Inverse Discrete Wavelet Transform (IDWT) module.

        This class implements a single-layer inverse wavelet transform for 1D signals based on a specified wavelet type
        (e.g., 'bior4.4'). It uses reconstruction filters (low-pass and high-pass) from PyWavelets to reconstruct the original
        signal from low-frequency (cA) and high-frequency (cD) coefficients via convolution.

        Args:
            wavelet (str): The wavelet type, defaults to 'bior4.4'. Must be a valid wavelet name supported by PyWavelets
                (e.g., 'db1', 'haar').
    """
    def __init__(self, wavelet: str = 'bior4.4'):
        """
            Initializes the InverseDWT1D module.

            Args:
                wavelet (str): The wavelet type, defaults to 'bior4.4'. Must be a valid wavelet name supported by PyWavelets.

            Attributes:
                rec_lo_filter (torch.Tensor): Low-pass reconstruction filter, reversed filter coefficients, shape (1, 1, kernel_len).
                rec_hi_filter (torch.Tensor): High-pass reconstruction filter, reversed filter coefficients, shape (1, 1, kernel_len).
                padding (int): Total padding required for convolution, equal to filter length minus 1.
                pad (int): Single-side padding length, equal to (filter length - 1) // 2.

            Example:
                >>> idwt = InverseDWT1D(wavelet='db4')
        """
        super().__init__()
        self.wavelet = wavelet

        w = pywt.Wavelet(wavelet)
        rec_lo = np.array(w.rec_lo)
        rec_hi = np.array(w.rec_hi)

        kernel_len = len(rec_lo)

        rec_lo = np.array(w.rec_lo)
        rec_hi = np.array(w.rec_hi)


        self.register_buffer('rec_lo_filter', torch.tensor(rec_lo[::-1].copy()).view(1, 1, len(rec_lo)).float())
        self.register_buffer('rec_hi_filter', torch.tensor(rec_hi[::-1].copy()).view(1, 1, len(rec_hi)).float())

        self.padding = kernel_len - 1
        self.pad = (kernel_len - 1) // 2

    def forward(self, cA: torch.Tensor, cD: torch.Tensor) -> torch.Tensor:
        """
            Performs a one-dimensional inverse discrete wavelet transform.

            Reconstructs the original signal by upsampling and filtering the low-frequency (cA) and high-frequency (cD) coefficients.
            The input tensors cA and cD must have the same shape (batch_size, length).

            Args:
                cA (torch.Tensor): Low-frequency coefficients (approximation), shape (batch_size, length).
                cD (torch.Tensor): High-frequency coefficients (detail), shape (batch_size, length).

            Returns:
                torch.Tensor: Reconstructed signal, shape (batch_size, reconstructed_length).

            Example:
                >>> import torch
                >>> idwt = InverseDWT1D(wavelet='bior4.4')
                >>> cA = torch.randn(32, 128)  # Batch size 32, sequence length 128
                >>> cD = torch.randn(32, 128)
                >>> output = idwt(cA, cD)  # Output shape approximately (32, 256)
        """

        cA = cA.unsqueeze(1)  # (B, 1, L)
        cD = cD.unsqueeze(1)  # (B, 1, L)


        up_cA = F.interpolate(cA, scale_factor=2, mode='nearest')
        up_cD = F.interpolate(cD, scale_factor=2, mode='nearest')


        x_lo = F.conv1d(up_cA, self.rec_lo_filter, padding=self.pad)
        x_hi = F.conv1d(up_cD, self.rec_hi_filter, padding=self.pad)
        min_len = min(x_lo.shape[-1], x_hi.shape[-1])
        x_lo = x_lo[..., :min_len]
        x_hi = x_hi[..., :min_len]
        x = x_lo + x_hi

        return x.squeeze(1)



class MultiLevelIDWT1D(nn.Module):
    """
        A multi-level one-dimensional Inverse Discrete Wavelet Transform module.

        This class performs a multi-level inverse wavelet transform by stacking multiple `InverseDWT1D` layers to reconstruct
        the original signal from a set of wavelet coefficients (one low-frequency coefficient cA and multiple high-frequency
        coefficients cD). Supports specifying the number of levels and the target output signal length.

        Args:
            wavelet (str): The wavelet type, defaults to 'bior4.4'. Must be a valid wavelet name supported by PyWavelets.
            level (int): Number of inverse transform levels, defaults to 3.
            gene_dim (int): Target length of the output signal, defaults to 3000. The reconstructed signal is cropped or padded
                to this length.
    """
    def __init__(self, wavelet: str = 'bior4.4', level: int = 3, gene_dim: int = 3000):
        """
            Initializes the MultiLevelIDWT1D module.

            Args:
                wavelet (str): The wavelet type, defaults to 'bior4.4'. Must be a valid wavelet name supported by PyWavelets.
                level (int): Number of inverse transform levels, defaults to 3. Indicates the number of cD coefficients required.
                gene_dim (int): Target length of the output signal, defaults to 3000.

            Attributes:
                idwt_layers (nn.ModuleList): List of `level` `InverseDWT1D` layers for sequential reconstruction.

            Example:
                >>> multi_idwt = MultiLevelIDWT1D(wavelet='db4', level=2, gene_dim=3000)
        """
        super().__init__()
        self.level = level
        self.wavelet = wavelet
        self.gene_dim = gene_dim
        self.idwt_layers = nn.ModuleList([
            InverseDWT1D(wavelet) for _ in range(level)
        ])

    def forward(self, coeffs: List[torch.Tensor]) -> torch.Tensor:
        """
            Performs a multi-level one-dimensional inverse discrete wavelet transform.

            Takes a list of wavelet coefficients (one low-frequency coefficient cA and multiple high-frequency coefficients cD)
            and reconstructs the original signal by applying `InverseDWT1D` layer by layer. The length of the input coefficients
            list must be `level + 1`.

            Args:
                coeffs (List[torch.Tensor]): List of wavelet coefficients, containing one low-frequency coefficient (cA) and
                    `level` high-frequency coefficients (cD). Each tensor has shape (batch_size, length). The first element is cA,
                    followed by cD1, cD2, etc.

            Returns:
                torch.Tensor: Reconstructed signal, shape (batch_size, gene_dim).

            Raises:
                AssertionError: If the length of the `coeffs` list is not equal to `level + 1`.

            Example:
                >>> import torch
                >>> multi_idwt = MultiLevelIDWT1D(wavelet='bior4.4', level=2, gene_dim=1000)
                >>> coeffs = [torch.randn(32, 128), torch.randn(32, 128), torch.randn(32, 256)]
                >>> output = multi_idwt(coeffs)  # Output shape (32, 1000)
        """

        assert len(
            coeffs) == self.level + 1, f"Expected {self.level + 1} coefficients (1 cA + {self.level} cD), got {len(coeffs)}"

        x = coeffs[0]  #
        for i in range(self.level):
            cD = coeffs[i + 1]
            x = self.idwt_layers[i](x, cD)

        return x[:, :self.gene_dim]





