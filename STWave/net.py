import torch
from torch.nn.modules.module import Module
from .layers import GAT, GAT_noncluster, CrossAttention
from .utils import select_device
from .wavetrans import Wavelet1DEncoderDecoder
from .inverseWT import MultiLevelIDWT1D
import os


class STWaveNet(Module):
    """
        Spatial-Temporal Wavelet Network (STWaveNet) for graph-structured data processing.

        This class integrates wavelet transformation, graph attention mechanisms, and cross-attention to process
        graph-structured data with spatial and temporal components. It supports both clustered (pruned) and non-clustered
        graph processing, using wavelet-based encoding/decoding and graph attention networks (GAT).

        The module can operate in two modes: standard (with input data `X`) or big (with precomputed wavelet coefficients).
        It applies wavelet transformation, processes features with GAT or GAT_noncluster, and combines results using cross-attention.

        Args:
            X (torch.Tensor, optional): Input data tensor, shape (batch_size, gene_dim). Required in standard mode (big=False).
            dim (int, optional): Input feature dimension. Required in both modes.
            dims_spot (List[int], optional): Hidden dimensions for GAT layers, defaults to [64, 16].
            dropout (float, optional): Dropout probability, defaults to 0.1.
            device (torch.device, optional): Device to run the model on. If None, automatically selected using `select_device`.
            cluster (bool, optional): Whether to use clustered (pruned) graph processing, defaults to True.
            wavelet (str, optional): Wavelet type for transformation, defaults to 'bior4.4' if None.
            level (int, optional): Number of wavelet decomposition levels, defaults to 3 if None.
            big (bool, optional): If True, operates in big mode with precomputed wavelet coefficients, defaults to False.
            wave_coff_dims (List[Tuple[int, int]], optional): Wavelet coefficient dimensions, required in big mode.
    """
    def __init__(self, X=None, dim=None, dims_spot=[64, 16], dropout=0.1, device=None, cluster=True,
                 wavelet: str = None, level: int = None, big=False, wave_coff_dims=None):
        """
            Initializes the STWaveNet module.

            Args:
                X (torch.Tensor, optional): Input data tensor, shape (batch_size, gene_dim). Required in standard mode (big=False).
                dim (int, optional): Input feature dimension. Required in both modes.
                dims_spot (List[int], optional): Hidden dimensions for GAT layers, defaults to [64, 16].
                dropout (float, optional): Dropout probability, defaults to 0.1.
                device (torch.device, optional): Device to run the model on. If None, automatically selected using `select_device`.
                cluster (bool, optional): Whether to use clustered (pruned) graph processing, defaults to True.
                wavelet (str, optional): Wavelet type for transformation, defaults to 'bior4.4' if None.
                level (int, optional): Number of wavelet decomposition levels, defaults to 3 if None.
                big (bool, optional): If True, operates in big mode with precomputed wavelet coefficients, defaults to False.
                wave_coff_dims (List[Tuple[int, int]], optional): Wavelet coefficient dimensions, required in big mode.

            Attributes:
                device (torch.device): Device on which the model runs.
                dropout (float): Dropout probability.
                NFeature (int): Input feature dimension.
                cluster (bool): Whether to use clustered graph processing.
                wavelet (str): Wavelet type used for transformation.
                level (int): Number of wavelet decomposition levels.
                wavetrans (Wavelet1DEncoderDecoder): Wavelet encoder/decoder (standard mode only).
                X_waveleted (torch.Tensor): Wavelet-transformed input data (standard mode only).
                wave (List[torch.Tensor]): List of wavelet coefficients (standard mode only).
                waveleted_dim (int): Dimension of wavelet-transformed data.
                recon_waveleted (torch.Tensor): Reconstructed wavelet coefficients (standard mode only).
                GATE_spot (Union[GAT, GAT_noncluster]): Graph attention network (clustered or non-clustered).
                CrossAtt (CrossAttention): Cross-attention module.
                iwat (MultiLevelIDWT1D): Multi-level inverse wavelet transform module.

            Raises:
                ValueError: If `X` or `dim` is None in standard mode, or if `wave_coff_dims` is None in big mode.

            Example:
                >>> import torch
                >>> X = torch.randn(100, 3000)
                >>> net = STWaveNet(X=X, dim=3000, dims_spot=[64, 16], cluster=True)
        """
        super(STWaveNet, self).__init__()
        if device is None:
            device = select_device()
            self.device = device
        if not big:
            if X == None or dim == None:
                raise ValueError('X is not None')
            self.dropout = dropout
            self.NFeature = dim
            self.cluster = cluster
            self.X = X.to(device)
            if wavelet is None:
                self.wavelet = 'bior4.4'
            else:
                self.wavelet = wavelet
            print('using wavelet:', self.wavelet)
            if level is None:
                self.level = 3
            else:
                self.level = level
            print('deconposing levels:', self.level)
            self.wavetrans = Wavelet1DEncoderDecoder(self.X, self.wavelet, self.level)
            self.X_waveleted = self.wavetrans.encode(self.X).to(self.device)
            self.wave = []

            for i, j in self.wavetrans.coeff_slices:
                self.wave.append(self.X_waveleted[:, i:j])
            self.waveleted_dim = self.X_waveleted.shape[1]
            self.recon_waveleted = torch.zeros_like(self.X_waveleted)
            dim = 0
            for i in range(self.level):
                dim += self.wavetrans.coeff_dims[i]

            self.recon_waveleted[:, :dim] = self.X_waveleted[:, :dim]
            self.recon_waveleted = self.wavetrans.decode(self.recon_waveleted).to(self.device)
            if cluster:
                self.GATE_spot = GAT(hidden_dims=[self.waveleted_dim] + dims_spot).to(device)
            else:
                self.GATE_spot = GAT_noncluster(hidden_dims=[self.waveleted_dim] + dims_spot).to(device)
            self.CrossAtt = CrossAttention(dim=self.NFeature, dropout=0.0)
            self.iwat = MultiLevelIDWT1D(wavelet=self.wavelet, level=self.level, gene_dim=self.NFeature)
        else:
            if wave_coff_dims is None:
                raise ValueError('wave_coff_dims needs to be provided')
            self.coff_dims = wave_coff_dims
            self.dropout = dropout
            if dim is None:
                raise ValueError('dim is not None')
            self.NFeature = dim
            self.cluster = cluster
            if wavelet is None:
                self.wavelet = 'bior4.4'
            else:
                self.wavelet = wavelet
            print('using wavelet:', self.wavelet)
            if level is None:
                self.level = 3
            else:
                self.level = level
            print('deconposing levels:', self.level)
            self.waveleted_dim = wave_coff_dims[-1][-1]
            if cluster:
                self.GATE_spot = GAT(hidden_dims=[self.waveleted_dim] + dims_spot).to(device)
            else:
                self.GATE_spot = GAT_noncluster(hidden_dims=[self.waveleted_dim] + dims_spot).to(device)
            self.CrossAtt = CrossAttention(dim=self.NFeature, dropout=0.0)
            self.iwat = MultiLevelIDWT1D(wavelet=self.wavelet, level=self.level, gene_dim=self.NFeature)

    def forward(self, edge_index, edge_index_prun=None):
        """
            Forward pass of the STWaveNet module (standard mode).

            Processes wavelet-transformed input data through a GAT or GAT_noncluster module, reconstructs the signal
            using inverse wavelet transform, and combines the results with cross-attention.

            Args:
                edge_index (torch.Tensor): Edge indices for the graph, shape (2, num_edges).
                edge_index_prun (torch.Tensor, optional): Pruned edge indices for clustered graph processing,
                    shape (2, num_edges_prun). Required if cluster=True.

            Returns:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                    - res: Combined feature representation, shape (batch_size, NFeature).
                    - lamba: Attention weights from cross-attention, shape (batch_size, 2).
                    - emb_spot: Intermediate GAT representation, shape (batch_size, out_dim).
                    - mtx_res_spot: Final GAT representation, shape (batch_size, waveleted_dim).

            Example:
                >>> import torch
                >>> X = torch.randn(100, 3000)
                >>> net = STWaveNet(X=X, dim=3000, cluster=True)
                >>> edge_index = torch.randint(0, 100, (2, 600))
                >>> edge_index_prun = torch.randint(0, 100, (2, 420))
                >>> res, lamba, emb_spot, mtx_res_spot = net(edge_index, edge_index_prun)
        """
        if self.cluster and edge_index_prun != None:
            emb_spot, mtx_res_spot = self.GATE_spot(self.X_waveleted, edge_index, edge_index_prun)
        else:
            emb_spot, mtx_res_spot = self.GATE_spot(self.X_waveleted, edge_index)

        X_res_net = self.iwat([mtx_res_spot[:, i:j] for i, j in self.wavetrans.coeff_slices])
        lamba, res = self.CrossAtt(X_res_net, self.recon_waveleted)
        return res, lamba, emb_spot, mtx_res_spot

    def big_forward(self, x, decode_waved, edge_index, edge_index_prun=None):
        """
            Forward pass of the STWaveNet module (big mode).

            Processes precomputed wavelet-transformed input data through a GAT or GAT_noncluster module, reconstructs
            the signal using inverse wavelet transform, and combines the results with cross-attention.

            Args:
                x (torch.Tensor): Wavelet-transformed input data, shape (batch_size, waveleted_dim).
                decode_waved (torch.Tensor): Decoded wavelet data for cross-attention, shape (batch_size, NFeature).
                edge_index (torch.Tensor): Edge indices for the graph, shape (2, num_edges).
                edge_index_prun (torch.Tensor, optional): Pruned edge indices for clustered graph processing,
                    shape (2, num_edges_prun). Required if cluster=True.

            Returns:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                    - res: Combined feature representation, shape (batch_size, NFeature).
                    - lamba: Attention weights from cross-attention, shape (batch_size, 2).
                    - emb_spot: Intermediate GAT representation, shape (batch_size, out_dim).
                    - mtx_res_spot: Final GAT representation, shape (batch_size, waveleted_dim).

            Example:
                >>> import torch
                >>> net = STWaveNet(dim=3000, big=True, wave_coff_dims=[(0, 100), (100, 200), (200, 400)])
                >>> x = torch.randn(100, 400)
                >>> decode_waved = torch.randn(100, 3000)
                >>> edge_index = torch.randint(0, 100, (2, 600))
                >>> res, lamba, emb_spot, mtx_res_spot = net.big_forward(x, decode_waved, edge_index)
        """
        if self.cluster and edge_index_prun != None:
            emb_spot, mtx_res_spot = self.GATE_spot(x, edge_index, edge_index_prun)
        else:
            emb_spot, mtx_res_spot = self.GATE_spot(x, edge_index)
        X_res_net = self.iwat([mtx_res_spot[:, i:j] for i, j in self.coff_dims])
        lamba, res = self.CrossAtt(X_res_net, decode_waved)
        return res, lamba, emb_spot, mtx_res_spot


    def save_model_weifgt(self, path):
        """
            Saves the model's state dictionary to a file.

            Creates the directory if it does not exist and ensures the file has a `.pth` extension.

            Args:
                path (str): Path to save the model weights.

            Raises:
                Exception: If an error occurs during saving.

            Example:
                >>> net = STWaveNet(X=torch.randn(100, 3000), dim=3000)
                >>> net.save_model_weifgt('model_weights.pth')
        """
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


        if not path.endswith('.pth'):
            path += '.pth'

        try:
            torch.save(self.state_dict(), path)
            print(f"The model parameters have been successfully saved: {path}")
        except Exception as e:
            print(f"An error occurred when saving the model parameters: {e}")
            raise e

    def load_model_weight(self, path):
        """
            Loads the model's state dictionary from a file.

            Loads the weights onto the model's device and updates the model's state.

            Args:
                path (str): Path to the model weights file.

            Raises:
                FileNotFoundError: If the specified file does not exist.
                Exception: If an error occurs during loading.

            Example:
                >>> net = STWaveNet(X=torch.randn(100, 3000), dim=3000)
                >>> net.load_model_weight('model_weights.pth')
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"The model file does not exist: {path}")

            state_dict = torch.load(path, map_location=self.device)
            self.load_state_dict(state_dict)
            print(f"The model parameters have been successfully obtained from {path} loading")
        except Exception as e:
            print(f"An error occurred while loading the model parameters: {e}")
            raise e


if __name__ == '__main__':
    import numpy as np

    torch.manual_seed(42)
    np.random.seed(42)
    batch_size, gene_num = 100, 3000
    device = select_device()
    expression_data = torch.randn(batch_size, gene_num).abs()
    expression_data = expression_data.to(device)
    num_nodes = 100
    num_edges = 600
    prune_ratio = 0.3
    edge_index = torch.zeros(2, num_edges, dtype=torch.long)
    for i in range(num_edges):
        src = torch.randint(0, num_nodes, (1,)).item()
        dst = torch.randint(0, num_nodes, (1,)).item()
        while dst == src or (dst, src) in edge_index[:, :i].t().tolist():
            dst = torch.randint(0, num_nodes, (1,)).item()
        edge_index[0, i] = src
        edge_index[1, i] = dst


    num_edges_prun = int(num_edges * (1 - prune_ratio))
    prune_indices = torch.randperm(num_edges)[:num_edges_prun]
    edge_index_prun = edge_index[:, prune_indices]
    edge_index = edge_index.to(device)
    edge_index_prun = edge_index_prun.to(device)
    net = STWaveNet(expression_data, dim=gene_num)
    net.to(device)
    net(edge_index, edge_index_prun)

    print('finished')


