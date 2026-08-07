import numpy as np
import scipy.sparse
import random
from tqdm import tqdm
import warnings
from .net import STWaveNet
from .utils import Transfer_Graph_Data
import torch
import time
import torch.backends.cudnn as cudnn
cudnn.enable = True
cudnn.benchmark = True
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.loader import NeighborLoader
from .wavetrans import Wavelet1DEncoderDecoder
from typing import Literal

def check_data_devices(data):
    """
        Prints the device, shape, and data type of tensor attributes in a PyTorch Geometric Data object.

        Args:
            data (Data): PyTorch Geometric Data object containing graph data.

        Example:
            >>> from torch_geometric.data import Data
            >>> data = Data(x=torch.randn(100, 64), edge_index=torch.randint(0, 100, (2, 200)))
            >>> check_data_devices(data)
    """
    print("check PyG Data object device：\n")
    for key, value in data:
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={tuple(value.shape)}, device={value.device}, dtype={value.dtype}")
        else:
            print(f"{key}: no tensor type, type is {type(value)}")


class Trainer:
    """
        Trainer class for training and inference with the STWaveNet model.

        This class handles the preparation, training, and inference of the STWaveNet model on spatial transcriptomics data.
        It supports dimensional reduction, batch processing, and wavelet transformation for graph-structured data.

        Args:
            adata (AnnData): AnnData object containing spatial transcriptomics data.
            dim_reduction (str, optional): Dimensional reduction method ('PCA', 'HVG', or None). Defaults to 'PCA'.
            batch_data (bool, optional): Whether to split data into batches for training, defaults to False.
            num_batch_x_y (Tuple[int, int], optional): Number of batches along x and y axes if batch_data=True, defaults to None.
            device_idx (int, optional): Index of the CUDA device to use, defaults to 0.
            verbose (bool, optional): If True, prints additional information during processing, defaults to False.
            center_msg (str, optional): Message passing mode for graph data ('in' or 'out'), defaults to 'out'.
            wavelet (str, optional): Wavelet type for transformation, defaults to 'db4'.
            level (int, optional): Number of wavelet decomposition levels, defaults to 3.
            weight (Dict[str, float], optional): Weights for reconstruction and wavelet losses, defaults to {'w_recon': 1, 'w_wave': 20}.

        Raises:
            ValueError: If required keys ('Spatial_Net', 'Precluster_Net', 'X_pca', or 'highly_variable') are missing in adata.
    """
    def __init__(self,
                 adata,
                 dim_reduction='PCA',
                 batch_data=False,
                 num_batch_x_y=None,
                 device:Literal['cuda', 'cpu']='cuda',
                 device_idx=0,
                 verbose=False,
                 center_msg='out',
                 wavelet='db4',
                 level=3,
                 weight=None):
        """
            Initializes the Trainer for STWaveNet.

            Args:
                adata (AnnData): AnnData object containing spatial transcriptomics data.
                dim_reduction (str, optional): Dimensional reduction method ('PCA', 'HVG', or None). Defaults to 'PCA'.
                batch_data (bool, optional): Whether to split data into batches for training, defaults to False.
                num_batch_x_y (Tuple[int, int], optional): Number of batches along x and y axes if batch_data=True, defaults to None.
                device ('cuda', 'cpu'): 'cuda' or 'cpu'.
                device_idx (int, optional): Index of the CUDA device to use, defaults to 0.
                verbose (bool, optional): If True, prints additional information during processing, defaults to False.
                center_msg (str, optional): Message passing mode for graph data ('in' or 'out'), defaults to 'out'.
                wavelet (str, optional): Wavelet type for transformation, defaults to 'db4'.
                level (int, optional): Number of wavelet decomposition levels, defaults to 3.
                weight (Dict[str, float], optional): Weights for reconstruction and wavelet losses, defaults to {'w_recon': 1, 'w_wave': 20}.

            Attributes:
                data (Data): PyTorch Geometric Data object containing graph data.
                wavetrans (Wavelet1DEncoderDecoder): Wavelet encoder/decoder for data transformation.
                device (torch.device): Device for computation (CPU or CUDA).
                model (STWaveNet): STWaveNet model instance.
                loader (DataLoader): DataLoader for batch processing (if batch_data=True).
                batch_index_list (List[torch.Tensor]): List of batch indices (if batch_data=True).

            Raises:
                ValueError: If 'Spatial_Net', 'Precluster_Net', 'X_pca' (for PCA), or 'highly_variable' (for HVG) are missing.

            Example:
                >>> import scanpy as sc
                >>> adata = sc.read_h5ad('data.h5ad')
                >>> trainer = Trainer(adata, dim_reduction='PCA', wavelet='bior4.4', level=3)
        """
        if dim_reduction == 'PCA':
            if 'X_pca' not in adata.obsm.keys():
                raise ValueError("PCA has not been done! Run sc.pp.pca first!")
        elif dim_reduction == 'HVG':
            if 'highly_variable' not in adata.var.keys():
                raise ValueError("HVG has not been computed! Run sc.pp.highly_variable_genes first!")
        else:
            # Warn that no valid dimensional reduction method was specified
            warnings.warn("No dimentional reduction method specified, using all genes' expression as input.")
        self.dim_reduction = dim_reduction  
        self.batch_data = batch_data  
        self.adata = adata  
        self.wavelet = wavelet
        self.level = level

        if 'Spatial_Net' not in adata.uns.keys():
            raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
        if 'Precluster_Net' not in adata.uns.keys():
            raise ValueError("Exp_Net is not existed! Run Cal_Expression_Net first!")

        # Prepare the graph data for training
        self.data = Transfer_Graph_Data(adata, dim_reduction=dim_reduction, center_msg=center_msg)
        check_data_devices(self.data)
        if verbose:
            print('Size of Input: ', self.data.x.shape)
        self.wavetrans = Wavelet1DEncoderDecoder(self.data.x, self.wavelet, self.level)
        self.data.waved = self.wavetrans.encode(self.data.x)
        coff = torch.zeros_like(self.data.waved)
        coff[:, :self.wavetrans.coeff_dims[0]] = self.data.waved[:, :self.wavetrans.coeff_dims[0]]
        self.data.wavedecode = self.wavetrans.decode(coff, is_nonoe_negetive=False)
        if device == 'cpu':
            self.device = device
        else:
            self.device = torch.device(f'cuda:{device_idx}' if torch.cuda.is_available() else 'cpu')
        print(f'------Using device: {self.device}')
        self.model = None
        if weight is None:
            weight = {'w_recon': 1, 'w_wave': 20}
        self.weight = weight
        print('------Using default weights: ', weight)

        if batch_data:
            self.num_batch_x, self.num_batch_y = num_batch_x_y
            self.batch_index_list = self.get_batch_index(self.num_batch_x, self.num_batch_y)
            self.data_list = self.sample_subgraph_from_batch_indices()
            self.loader = DataLoader(self.data_list, batch_size=1, shuffle=True)


    def train(self, save_path=None, hidden_dims=[100, 32], n_epochs=200, lr=0.0001,
              key_added='STWave', att_drop=0.3, gradient_clipping=5., weight_decay=0.0001, cluster=True,
              random_seed=2025, save_loss=False, save_reconstrction=True, batch_inference=True,):
        """
            Trains the STWaveNet model.

            Performs training with a combination of reconstruction and wavelet loss, optionally using batch processing.
            Saves model weights, loss values, and reconstructed data if specified.

            Args:
                save_path (str, optional): Directory to save model weights, defaults to None.
                hidden_dims (List[int], optional): Hidden dimensions for STWaveNet, defaults to [100, 32].
                n_epochs (int, optional): Number of training epochs, defaults to 200.
                lr (float, optional): Learning rate for the Adam optimizer, defaults to 0.0001.
                key_added (str, optional): Key to store embeddings in adata.obsm, defaults to 'STWave'.
                att_drop (float, optional): Dropout rate for attention layers, defaults to 0.3.
                gradient_clipping (float, optional): Gradient clipping threshold, defaults to 5.0.
                weight_decay (float, optional): Weight decay for the Adam optimizer, defaults to 0.0001.
                cluster (bool, optional): Whether to use clustered graph processing, defaults to True.
                random_seed (int, optional): Random seed for reproducibility, defaults to 2025.
                save_loss (bool, optional): If True, saves loss values in adata.uns, defaults to False.
                save_reconstrction (bool, optional): If True, saves reconstructed data in adata.obsm, defaults to True.
                batch_inference (bool, optional): If True, performs batch inference, defaults to True.

            Example:
                >>> trainer = Trainer(adata, dim_reduction='PCA')
                >>> trainer.train(save_path='models', n_epochs=100, hidden_dims=[64, 16])
        """
        # Set the path where the model will be saved
        self.save_path = save_path
        # Set random seeds
        seed = random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Initialize the model
        if self.model is None:
            # Create a new STWave model if one does not exist
            model = STWaveNet(X=self.data.x,dim=self.data.x.shape[1], cluster=cluster, dims_spot=hidden_dims, dropout=att_drop,
                                big=True if self.batch_data else False, wavelet=self.wavelet, level=self.level,
                                wave_coff_dims=self.wavetrans.coeff_slices).to(self.device)
        else:
            # Use the existing model and move it to the specified device
            model = self.model.to(self.device)

        if self.batch_data:
            data = self.data.to('cpu')  # Load data to CPU if using batch data
        else:
            data = self.data.to(self.device)  # Load data to the specified device

        # Set up the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # List to keep track of loss values during training
        loss_list = []
        print('------training batch...')
        time.sleep(0.01)
        with tqdm(range(n_epochs)) as tq:
            for _ in tq:
                if self.batch_data:
                    # Loop through batches of data if DIC is enabled
                    self.loss = 0
                    self.loss_recon = 0
                    self.loss_wave = 0
                    for batch in self.loader:
                        batch = batch.to(self.device)
                        model.train()
                        optimizer.zero_grad()
                        if not cluster:
                            res, _, emb_spot, decode_waved = model.big_forward(batch.waved, batch.wavedecode,
                                                                               batch.edge_index[:,
                                                                               batch.edge_type == 1])
                        else:
                            res, _, emb_spot, decode_waved = model.big_forward(batch.waved, batch.wavedecode,
                                                                               batch.edge_index[:,
                                                                               batch.edge_type == 1],
                                                                               batch.edge_index[:,
                                                                               batch.edge_type == 0])

                        res = res[:batch.batch_size]
                        decode_waved = decode_waved[:batch.batch_size]

                        loss_recon = F.mse_loss(res, batch.x[:batch.batch_size])
                        loss_wave = F.mse_loss(decode_waved, batch.waved[:batch.batch_size])

                        loss = self.weight['w_recon'] * loss_recon + self.weight['w_wave'] * loss_wave
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                        optimizer.step()
                        loss_list.append(loss.item())
                        self.loss += loss.item()
                        self.loss_recon += loss_recon.item()
                        self.loss_wave += loss_wave.item()
                    batch_n = len(self.loader)
                    tq.set_postfix(_1loss=self.loss / batch_n,
                                   _2loss_recon=self.loss_recon / batch_n,
                                   _3loss_wave=self.loss_wave / batch_n)
                    tq.update(1)

                else:
                    # Training without DIC
                    model.train()
                    optimizer.zero_grad()
                    res, lamba, emb_spot, decode_waved = model.forward(
                                                                           data.edge_index[:, data.edge_type == 1],
                                                                           data.edge_index[:, data.edge_type == 0])
                    loss_recon = F.mse_loss(res, data.x)
                    loss_wave = F.mse_loss(decode_waved, data.waved)
                    loss = self.weight['w_recon'] * loss_recon + self.weight['w_wave'] * loss_wave
                    loss.backward()
                    loss_list.append(loss.item())
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    optimizer.step()
                    tq.set_postfix(_1loss=loss.item(),
                                   _2loss_recon=loss_recon.item(),
                                   _3loss_wave=loss_wave.item())
        # Evaluation phase after training
        print('------inferencing...')
        time.sleep(0.01)
        with torch.no_grad():
            if self.batch_data:
                if batch_inference:
                    # sampling a batch for inference
                    model.to(self.device)
                    model.eval()
                    loader = self.batch_inference_sample(num_neighbors=[-1, -1])
                    res_list, _, emb_spot_list, _ = [], [], [], []
                    with tqdm(range(self.num_batch_x * self.num_batch_y)) as t:
                        for batch in self.data_list:
                            # for batch in loader:
                            batch = batch.to(self.device)
                            res, _, emb_spot, _ = model.big_forward(batch.waved, batch.wavedecode,
                                                                    batch.edge_index[:, batch.edge_type == 1],
                                                                    batch.edge_index[:, batch.edge_type == 0])

                            res = res[:batch.batch_size]
                            emb_spot = emb_spot[:batch.batch_size]
                            res_list.append(res.cpu())
                            emb_spot_list.append(emb_spot.cpu())
                            t.update(1)
                            batch = batch.to('cpu')

                    emb_spot = torch.zeros([self.data.x.shape[0], hidden_dims[-1]])
                    for e, i in zip(emb_spot_list, self.batch_index_list):
                        emb_spot[i] = e
                    # emb_spot = torch.cat(emb_spot_list, dim=0)
                    res = torch.zeros(self.data.x.shape)
                    for r, i in zip(res_list, self.batch_index_list):
                        res[i] = r
                    # res = torch.cat(res_list, dim=0)

                else:
                    # Move model to CPU for evaluation if using DIC
                    model.to('cpu')
                    model.eval()
                    res, _, emb_spot, _ = model.big_forward(data.waved, data.wavedecode,
                                                            data.edge_index[:, data.edge_type == 1],
                                                            data.edge_index[:, data.edge_type == 0])
                    model.to(self.device)
            else:
                model.eval()
                res, _, emb_spot, decode_waved = model.forward(#data.waved, data.wavedecode,
                                                                   data.edge_index[:, data.edge_type == 1],
                                                                   data.edge_index[:, data.edge_type == 0])
        # Store the STWave representations in the AnnData object
        tqdm.write('------saving embeddings...')
        STWave_rep = emb_spot.to('cpu').detach().numpy()
        self.adata.obsm[key_added] = STWave_rep

        # Save the trained model if a save path is provided
        if save_path is not None:
            tqdm.write('------saving model weight...')
            torch.save(model, f'{save_path}/model.pth')

        # Save loss values if requested
        if save_loss:
            tqdm.write('------saving losses...')
            self.adata.uns['STWave_loss'] = loss_list

        # Save reconstructed output if requested
        if save_reconstrction:
            tqdm.write('------saving reconstrction...')
            ReX = res.to('cpu').numpy()
            if self.dim_reduction != 'PCA':
                # idx = np.where(self.adata.X == 0)
                dense_X = self.adata.X.toarray() if scipy.sparse.issparse(self.adata.X) else self.adata.X
                idx = np.where(dense_X == 0)
                ReX[idx] = 0
            self.adata.obsm['STWave_ReX'] = ReX

        self.model = model

    def get_batch_index(self, num_batch_x, num_batch_y):
        """
            Generates batch indices based on spatial coordinates.

            Splits the spatial data into batches based on x and y coordinates.

            Args:
                num_batch_x (int): Number of batches along the x-axis.
                num_batch_y (int): Number of batches along the y-axis.

            Returns:
                List[torch.Tensor]: List of tensors containing indices for each batch.

            Example:
                >>> trainer = Trainer(adata, batch_data=True, num_batch_x_y=(2, 2))
                >>> batch_indices = trainer.get_batch_index(2, 2)
        """
        Sp_df = self.adata.obsm['spatial']
        # Calculate the x-coordinates for the specified number of batches
        batch_x_coor = np.percentile(Sp_df[:, 0], np.linspace(0, 100, num_batch_x + 1))
        # Calculate the y-coordinates for the specified number of batches
        batch_y_coor = np.percentile(Sp_df[:, 1], np.linspace(0, 100, num_batch_y + 1))

        # Initialize an empty list to store each batch of data
        Batch_index_list = []

        # Iterate over the number of batches along the x-axis
        tqdm.write('------calculating batch indices...')
        with tqdm(range(num_batch_x * num_batch_y)) as t:
            for it_x in range(num_batch_x):
                # Get the min and max x-coordinates for the current batch
                min_x, max_x = batch_x_coor[it_x], batch_x_coor[it_x + 1]

                # Iterate over the number of batches along the y-axis
                for it_y in range(num_batch_y):
                    # Get the min and max y-coordinates for the current batch
                    min_y, max_y = batch_y_coor[it_y], batch_y_coor[it_y + 1]

                    # Create a mask for the x-coordinate to filter the data
                    mask_x = (Sp_df[:, 0] >= min_x) & (Sp_df[:, 0] <= max_x)
                    # Create a mask for the y-coordinate to filter the data
                    mask_y = (Sp_df[:, 1] >= min_y) & (Sp_df[:, 1] <= max_y)
                    # Combine both masks to get the final mask
                    mask = mask_x & mask_y
                    indices = torch.from_numpy(np.where(mask)[0])
                    indices = indices.to('cpu')

                    # Check if the batch contains more than 10 spots
                    if indices.shape[0] > 10:
                        Batch_index_list.append(indices)  # Add the valid batch to the list
                    t.update(1)  # Update the progress bar

        # If plot_Stats is True, visualize the distribution of spots per batch

        return Batch_index_list  # Return the list of batches

    def sample_subgraph_from_batch_indices(self, num_neighbors_per_layer=[-1, -1]):
        """
            Samples subgraphs for each batch using NeighborLoader.

            Args:
                num_neighbors_per_layer (List[int], optional): Number of neighbors to sample per layer, defaults to [-1, -1].

            Returns:
                List[Data]: List of PyTorch Geometric Data objects for each batch.

            Example:
                >>> trainer = Trainer(adata, batch_data=True, num_batch_x_y=(2, 2))
                >>> subgraphs = trainer.sample_subgraph_from_batch_indices()
        """
        subgraph_list = []

      
        tqdm.write('------sampling subgraphs...')
        with tqdm(self.batch_index_list) as t:
            for batch_indices in self.batch_index_list:
                input_nodes = torch.tensor(batch_indices, dtype=torch.long)


                loader = NeighborLoader(
                    self.data,
                    input_nodes=input_nodes,
                    num_neighbors=num_neighbors_per_layer,
                    batch_size=len(input_nodes),
                    shuffle=False
                )


                sampled_batch = next(iter(loader))


                subgraph_list.append(sampled_batch)
                t.update(1)

        return subgraph_list

    def batch_inference_sample(self, num_neighbors=[16, 16], batch_size=1024, shuffle=False):
        """
            Creates a NeighborLoader for batch inference.

            Args:
                num_neighbors (List[int], optional): Number of neighbors to sample per layer, defaults to [16, 16].
                batch_size (int, optional): Batch size for inference, defaults to 1024.
                shuffle (bool, optional): Whether to shuffle the data, defaults to False.

            Returns:
                NeighborLoader: PyTorch Geometric NeighborLoader for batch inference.

            Example:
                >>> trainer = Trainer(adata, batch_data=True)
                >>> loader = trainer.batch_inference_sample(num_neighbors=[10, 10], batch_size=512)
        """
        data = self.data
        loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=None,
            shuffle=shuffle
        )
        return loader

    def load_model(self, path):
        """
            Loads a trained STWaveNet model from a file.

            Args:
                path (str): Path to the saved model file.

            Example:
                >>> trainer = Trainer(adata, dim_reduction='PCA')
                >>> trainer.load_model('models/model.pth')
        """

        self.model = torch.load(path, map_location=self.device)

    def save_model(self, path):
        """
            Saves the trained STWaveNet model to a file.

            Args:
                path (str): Directory to save the model file.

            Example:
                >>> trainer = Trainer(adata, dim_reduction='PCA')
                >>> trainer.save_model('models')
        """

        torch.save(self.model, f'{path}/model.pth')

    @torch.no_grad()
    def process(self, gdata=None):
        """
            Performs inference using the STWaveNet model.

            Stores the resulting embeddings and reconstructed data in the AnnData object.

            Args:
                gdata (Data, optional): PyTorch Geometric Data object for inference. If None, uses the registered data.

            Example:
                >>> trainer = Trainer(adata, dim_reduction='PCA')
                >>> trainer.train(n_epochs=100)
                >>> trainer.process()
        """

        if gdata is None:
            gdata = self.data
        self.model.to('cpu')
        self.model.eval()
        gdata = gdata.to('cpu')
        res, _, emb_spot, _ = self.model.big_forward(gdata.x, gdata.wavedecode,
                                                     gdata.edge_index[:, gdata.edge_type == 1],
                                                     gdata.edge_index[:, gdata.edge_type == 0])
        STWave_rep = emb_spot.to('cpu').detach().numpy()
        self.adata.obsm['STWave'] = STWave_rep
        if self.dim_reduction != 'PCA':
            ReX = res.to('cpu').detach().numpy()
            idx = np.where(self.adata.X == 0)
            ReX[idx] = 0
            self.adata.obsm['STWave_ReX'] = ReX




