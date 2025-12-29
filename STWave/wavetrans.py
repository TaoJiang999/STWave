import torch
import torch.nn as nn
import pywt
import numpy as np
from .utils import select_device

class Wavelet1DEncoderDecoder(nn.Module):
    def __init__(self, expression_matrix, wavelet='bior4.4', level=3):
        """
        Encoder-decoding module based on one-dimensional discrete wavelet transform

        Parameters
        ----------
        expression_matrix : torch.Tensor
            Gene expression matrix, shape [B, G], where B represents the number of samples and G represents the number of genes
        wavelet : str
            The wavelet basis name must support DWT and inverse transformation (such as 'db4', 'bior4.4')
        level : int
            The number of wavelet decomposition layers
        """
        super(Wavelet1DEncoderDecoder, self).__init__()
        self.device = select_device()
        self.wavelet = wavelet
        self.level = level
        self.expression_matrix = expression_matrix.to(self.device)
        self.batch_size, self.gene_num = expression_matrix.shape

        self._initialize_wavelet_transform()

    def _initialize_wavelet_transform(self):
        """
        Initialize the wavelet transform and save the dimension information
        """
        sample = self.expression_matrix[0].detach().cpu().numpy()
        coeffs = pywt.wavedec(sample, self.wavelet, level=self.level)


        self.coeff_dims = [len(c) for c in coeffs]
        self.total_dim = sum(self.coeff_dims)


        self.coeff_slices = []
        start_idx = 0
        for dim in self.coeff_dims:
            end_idx = start_idx + dim
            self.coeff_slices.append((start_idx, end_idx))
            start_idx = end_idx

    def encode(self, x=None):
        """
        Batch one-dimensional wavelet transform encoding is performed on the expression matrix

        Parameters
        ----------
        x : torch.Tensor, optional
            Input the shape [B, G]. If it is None, use the expression_matrix at the time of initialization

        Returns
        -------
        encoded : torch.Tensor
            The encoded wavelet coefficients, shape [B, D], where D is the total length of all wavelet coefficients
        """
        if x is None:
            x = self.expression_matrix


        x_np = x.detach().cpu().numpy()


        encoded_list = []
        for sample in x_np:
            coeffs = pywt.wavedec(sample, self.wavelet, level=self.level)
            flat_coeffs = np.concatenate(coeffs)
            encoded_list.append(flat_coeffs)


        encoded = torch.tensor(np.stack(encoded_list), dtype=x.dtype).to(self.device)
        return encoded.to('cpu')

    def decode(self, encoded,is_nonoe_negetive=True):
        """
        Decode the encoded wavelet coefficients and reconstruct the gene expression matrix

        Parameters
        ----------
        encoded : torch.Tensor
            The encoded wavelet coefficients, shape [B, D]

        Returns
        -------
        decoded : torch.Tensor
            The reconstructed gene expression matrix, with the shape [B, G], is guaranteed to be non-negative
        """

        encoded_np = encoded.detach().cpu().numpy()

        decoded_list = []
        for sample in encoded_np:

            coeffs = []
            for start_idx, end_idx in self.coeff_slices:
                coeffs.append(sample[start_idx:end_idx])


            reconstructed = pywt.waverec(coeffs, self.wavelet)


            if len(reconstructed) > self.gene_num:
                reconstructed = reconstructed[:self.gene_num]
            elif len(reconstructed) < self.gene_num:

                reconstructed = np.pad(reconstructed, (0, self.gene_num - len(reconstructed)), 'constant')

            decoded_list.append(reconstructed)


        decoded = torch.tensor(np.stack(decoded_list), dtype=encoded.dtype).to(self.device)
        if is_nonoe_negetive:

            decoded = torch.clamp(decoded, min=0.0)

        return decoded.to('cpu')

    def forward(self, x=None):
        """
        Forward propagation returns the encoded features

        Parameters
        ----------
        x : torch.Tensor, optional
            Input the shape [B, G]. If it is None, use the expression_matrix at the time of initialization

        Returns
        -------
        encoded : torch.Tensor
            The encoded wavelet coefficients, shape [B, D]
        """
        return self.encode(x)

    def get_encoded_dim(self):
        """
        Obtain the encoded feature dimensions

        Returns
        -------
        int
            The encoded feature dimension
        """
        return self.total_dim

    def get_coeff_dims(self):
        """
        Obtain the dimension information of each layer of the wavelet coefficient

        Returns
        -------
        list
            The dimension list of wavelet coefficients at each layer
        """
        return self.coeff_dims.copy()

    def encode_decode(self, x=None):
        """
        The complete encoding-decoding process

        Parameters
        ----------
        x : torch.Tensor, optional
            Input the shape [B, G]. If it is None, use the expression_matrix at the time of initialization

        Returns
        -------
        encoded : torch.Tensor
            The encoded wavelet coefficients, shape [B, D]
        decoded : torch.Tensor
            The reconstructed gene expression matrix, with the shape [B, G], is guaranteed to be non-negative
        """
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded



if __name__ == "__main__":

    batch_size, gene_num = 100, 3000
    device = select_device()
    expression_data = torch.randn(batch_size, gene_num).abs() 
    expression_data = expression_data.to(device)

    wavelet_model = Wavelet1DEncoderDecoder(
        expression_matrix=expression_data,
        wavelet='bior4.4',
        level=3
    )


    print(f"The number of original genes: {gene_num}")
    print(f"Encoded dimension: {wavelet_model.get_encoded_dim()}")
    print(f"Coefficient dimensions of each layer: {wavelet_model.get_coeff_dims()}")


    encoded = wavelet_model.encode()
    print(f"The shape after encoding: {encoded.shape}")


    decoded = wavelet_model.decode(encoded)
    print(f"The shape after decoding: {decoded.shape}")


    mse_loss = torch.nn.MSELoss().to(device)
    reconstruction_error = mse_loss(decoded, expression_data)
    print(f"Reconstruction Error (MSE): {reconstruction_error.item():.6f}")


    print(f"Whether all are non-negative after decoding: {torch.all(decoded >= 0).item()}")


    new_data = torch.randn(50, gene_num).abs()
    new_encoded = wavelet_model.encode(new_data)
    new_decoded = wavelet_model.decode(new_encoded)
    print(f"New data encoding and decoding shapes: {new_encoded.shape} -> {new_decoded.shape}")