
import random
import warnings
import numpy as np
import time
from tqdm import tqdm
import scipy.sparse as sp
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from torch_geometric.data import Data
import seaborn as sns
from matplotlib import pyplot as plt
import torch
from typing import Union
import scanpy as sc
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from typing import Literal
import ot
from .svg_select import rank_gene_smooth,low_pass_enhancement



def seed_everything(seed):
    """
        Set random seeds for reproducibility across multiple libraries.

        Initializes the random seed for PyTorch, NumPy, Python's random module, and CUDA to ensure consistent results.

        Args:
            seed (int): Seed value for random number generators.

        Example:
            >>> seed_everything(42)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # dgl.random.seed(seed)



def clear_warnings(func, category=FutureWarning):
    """
        Decorator to suppress specified warnings during function execution.

        Args:
            func (callable): The function to wrap.
            category (type): Warning category to suppress, defaults to FutureWarning.

        Returns:
            callable: Wrapped function that suppresses the specified warnings.

        Example:
            >>> @clear_warnings
            ... def some_function():
            ...     # Function that may raise warnings
            ...     pass
            >>> some_function()
    """
    def warp(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=category)
            temp = func(*args, **kwargs)
            return temp
    return warp


def select_device(GPU: Union[bool, str] = True,):
    """
        Selects the computation device (CPU or GPU) based on availability and input preference.

        Args:
            GPU (Union[bool, str]): If True, prefers GPU if available; if a string, specifies the GPU device (e.g., 'cuda:0').
                If False, uses CPU. Defaults to True.

        Returns:
            torch.device: Selected device ('cuda' or 'cpu').

        Example:
            >>> device = select_device(GPU=True)
            >>> print(device)
            cuda:0
    """
    if GPU:
        if torch.cuda.is_available():
            if isinstance(GPU, str):
                device = torch.device(GPU)
            else:
                device = torch.device('cuda:0')
        else:
            print("GPU isn't available, and use CPU to train Docs.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    return device
@clear_warnings
def prefilter_genes(adata, min_counts=None, max_counts=None, min_cells=3, max_cells=None):
    """
        Filters genes in an AnnData object based on count or cell thresholds.

        Args:
            adata (AnnData): AnnData object containing gene expression data.
            min_counts (int, optional): Minimum number of counts required for a gene. Defaults to None.
            max_counts (int, optional): Maximum number of counts allowed for a gene. Defaults to None.
            min_cells (int, optional): Minimum number of cells a gene must be expressed in, defaults to 3.
            max_cells (int, optional): Maximum number of cells a gene can be expressed in. Defaults to None.

        Raises:
            ValueError: If none of min_counts, max_counts, min_cells, or max_cells is provided.

        Example:
            >>> import scanpy as sc
            >>> adata = sc.read_h5ad('data.h5ad')
            >>> prefilter_genes(adata, min_cells=5)
    """
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp = np.asarray([True] * adata.shape[1], dtype=bool)
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, min_cells=min_cells)[0]) if min_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, max_cells=max_cells)[0]) if max_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, min_counts=min_counts)[0]) if min_counts is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, max_counts=max_counts)[0]) if max_counts is not None else id_tmp
    adata._inplace_subset_var(id_tmp)

def res_search_fixed_clus(adata, fixed_clus_count, increment=0.02):
    """
        Searches for the Leiden clustering resolution that yields a specified number of clusters.

        Args:
            adata (AnnData): AnnData object containing the data to cluster.
            fixed_clus_count (int): Desired number of clusters.
            increment (float, optional): Step size for resolution search, defaults to 0.02.

        Returns:
            float: Resolution value that achieves the desired number of clusters.

        Example:
            >>> import scanpy as sc
            >>> adata = sc.read_h5ad('data.h5ad')
            >>> resolution = res_search_fixed_clus(adata, fixed_clus_count=10)
            >>> print(resolution)
    """
    for res in sorted(list(np.arange(0.01, 2.5, increment)), reverse=True):
        sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        if count_unique_leiden == fixed_clus_count:
            break
    return res


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb', random_seed=2023):
    """
        Performs clustering using the mclust R package.

        Applies PCA (if necessary) to reduce dimensionality before clustering with mclust.

        Args:
            adata (AnnData): AnnData object containing the data to cluster.
            num_cluster (int): Number of clusters to identify.
            modelNames (str): Model name for mclust, defaults to 'EEE'.
            used_obsm (str): Key in adata.obsm for the embedding to use, defaults to 'emb'.
            random_seed (int): Random seed for reproducibility, defaults to 2023.

        Returns:
            AnnData: Updated AnnData object with 'mclust' column in adata.obs containing cluster labels.

        Example:
            >>> import scanpy as sc
            >>> adata = sc.read_h5ad('data.h5ad')
            >>> adata = mclust_R(adata, num_cluster=5)
    """
    np.random.seed(random_seed)
    from sklearn.decomposition import PCA
    dim = adata.obsm[used_obsm].shape[1]
    if dim > 50:
        dim = 50
    
    pca = PCA(n_components=dim, random_state=random_seed)
    embedding = pca.fit_transform(adata.obsm[used_obsm].copy())
    adata.obsm['emb_pca'] = embedding

    import rpy2.robjects as robjects
    robjects.r.library('mclust')

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    # res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm['emb_pca']), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

@clear_warnings
def svg(adata, svg_method:Literal['seurat', 'seurat_v3', 'cell_ranger', 'mix']='seurat_v3', n_top=3000, csvg=0.0001, smoothing=True):
    """
    Select spatially variable genes using six methods, including
    'seurat', 'seurat_v3', 'cell_ranger' and 'mix'.

    Args:
        adata: anndata
            AnnData object of scanpy package.
        svg_method: str, optional
            Methods for selecting spatially variable genes. Teh default is 'gft_top'.
        n_top: int, optional
            Number of spatially variable genes selected. The default is 3000.
        csvg: float, optional
            Smoothing coefficient of GFT for noise reduction. The default is 0.0001.
        smoothing: bool, optional
            Determine whether it is smooth for noise reduction. The default is True.

    Returns:
        adata: anndata
            AnnData object of scanpy package after choosing svgs and smoothing.
    """
    assert svg_method in ['seurat', 'seurat_v3', 'cell_ranger', 'mix']
    if svg_method == 'seurat_v3':
        adata.var_names_make_unique()
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=n_top)
        top_genes_set = set(
            adata.var.sort_values('highly_variable_rank').head(n_top).index
        )
        ordered_top = [g for g in adata.var_names if g in top_genes_set]
        # adata = adata[:, adata.var['highly_variable']]
        adata = adata[:, ordered_top]
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    elif svg_method == 'mix':
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=int(n_top / 2))
        seuv3_list = adata.var_names[adata.var['highly_variable']]
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        gene_df = rank_gene_smooth(adata,
                                   spatial_info=['array_row', 'array_col'],
                                   ratio_low_freq=1,
                                   ratio_high_freq=1,
                                   ratio_neighbors=1,
                                   filter_peaks=True,
                                   S=6)
        svg_list = gene_df.index[:(n_top - len(seuv3_list))]
        merged_gene_list = np.union1d(seuv3_list, svg_list)
        adata = adata[:, merged_gene_list]
        if smoothing:
            low_pass_enhancement(adata,
                                 ratio_low_freq=15,
                                 c=csvg,
                                 spatial_info='spatial',
                                 ratio_neighbors=0.3,
                                 inplace=True)
    else:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        if svg_method == 'seurat':
            sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=n_top)
            adata = adata[:, adata.var['highly_variable']]
        elif svg_method == 'cell_ranger':
            sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes=n_top)
            adata = adata[:, adata.var['highly_variable']]
        elif svg_method == 'gft' or svg_method == 'gft_top':
            gene_df = rank_gene_smooth(adata,
                                       spatial_info='spatial',
                                       ratio_low_freq=1,
                                       ratio_high_freq=1,
                                       ratio_neighbors=1,
                                       filter_peaks=True,
                                       S=6)
            if svg_method == 'gft':
                svg_list = gene_df[gene_df.cutoff_gft_score][gene_df.qvalue < 0.05].index.tolist()
            elif svg_method == 'gft_top':
                svg_list = gene_df.index[:n_top].tolist()
            adata = adata[:, svg_list]
            adata_raw = adata.copy()
            if smoothing:
                low_pass_enhancement(adata,
                                     ratio_low_freq=15,
                                     c=csvg,
                                     spatial_info='spatial',
                                     ratio_neighbors=0.3,
                                     inplace=True)
    if(svg_method in ['gft', 'gft_top']):
        return adata, adata_raw
    return adata



def torch2array(x):
    """
        Converts a list of PyTorch tensors to NumPy arrays.

        Args:
            x (List[torch.Tensor]): List of PyTorch tensors to convert.

        Returns:
            List[np.ndarray]: List of NumPy arrays.

        Example:
            >>> import torch
            >>> tensors = [torch.randn(3), torch.randn(4)]
            >>> arrays = torch2array(tensors)
    """
    array = []
    for i in x:
        array.append(i.cpu().detach().numpy())
    return array

def refine_label(adata, radius=50, key='label'):
    """
    Smooths clustering labels based on spatial proximity.

    Refines labels by assigning each cell the most common label among its neighbors within a specified radius.

    Args:
        adata (AnnData): AnnData object containing spatial coordinates and labels.
        radius (int, optional): Number of nearest neighbors to consider, defaults to 50.
        key (str, optional): Key in adata.obs for the labels to refine, defaults to 'label'.

    Returns:
        List[str]: Refined labels for each cell.

    Example:
        >>> import scanpy as sc
        >>> adata = sc.read_h5ad('data.h5ad')
        >>> refined_labels = refine_label(adata, radius=30, key='mclust')
    """
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    # adata.obs['label_refined'] = np.array(new_type)
    return new_type


def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=4, model:Literal['KNN','Radius'] = 'KNN', verbose=True):
    """
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata : AnnData
        AnnData object from the scanpy package containing spatial coordinates.
    rad_cutoff : float or None
        Radius cutoff used when model='Radius' to determine connectivity based on distance.
    k_cutoff : int
        The number of nearest neighbors when model='KNN'.
        This parameter affects resource usage (time and memory). Reducing this can make training more efficient, but may slightly impact performance.
    model : str
        The network construction model.
        When model=='Radius', spots are connected to others within the specified radius.
        When model=='KNN', each spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    None
        The spatial networks are saved in adata.uns['Spatial_Net'].
    """

    # Ensure the specified model is valid
    assert model in ['Radius', 'KNN']  
    start = time.time()  # Start timing the function execution
    # Print progress message if verbose mode is enabled
    if verbose:
        print('------Calculating spatial graph...')

    # Retrieve spatial coordinates from the AnnData object
    coor = adata.obsm['spatial']  
    num_cells = coor.shape[0]  # Get the number of cells

    # Construct the neighbor graph based on the chosen model
    if model == 'Radius':
        # Use radius-based neighbor search
        nbrs = NearestNeighbors(radius=rad_cutoff).fit(coor)  # Fit the model with spatial coordinates
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)  # Get neighbors within the radius
    elif model == 'KNN':
        # Use k-nearest neighbors search
        nbrs = NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)  # Fit the model with spatial coordinates
        distances, indices = nbrs.kneighbors(coor)  # Get the k nearest neighbors

    # Build the list of edges for the spatial network
    KNN_list = [
        (i, indices[i][j], distances[i][j])  # Create tuples of (cell index, neighbor index, distance)
        for i in range(num_cells)  # Iterate over each cell
        for j in range(len(indices[i]))  # Iterate over each neighbor for the current cell
        if distances[i][j] > 0  # Ensure that the distance is greater than zero (exclude self-loops)
    ]

    # Create a DataFrame from the list of edges
    KNN_df = pd.DataFrame(KNN_list, columns=['Cell1', 'Cell2', 'Distance'])

    # Map the indices back to the actual cell names
    id_cell_trans = np.array(adata.obs.index)  # Get the cell names from the AnnData object
    KNN_df['Cell1'] = id_cell_trans[KNN_df['Cell1']]  # Map Cell1 indices to names
    KNN_df['Cell2'] = id_cell_trans[KNN_df['Cell2']]  # Map Cell2 indices to names

    # Print summary statistics if verbose mode is enabled
    if verbose:
        end = time.time()  # End timing the function execution
        print(f'Spatial graph contains {KNN_df.shape[0]} edges, {adata.n_obs} cells.')  # Number of edges and cells
        print(
            f'{KNN_df.shape[0] / adata.n_obs:.4f} neighbors per cell on average.')  # Average number of neighbors per cell
        print(f'Spatial graph calculation time: {end - start:.2f} seconds')
    # Save the spatial network DataFrame into the AnnData object
    adata.uns['Spatial_Net'] = KNN_df  # Store the constructed spatial network

def prune_spatial_Net(Graph_df, label):
    """
        Prunes a spatial network to keep only edges between cells with the same label.

        Args:
            Graph_df (pd.DataFrame): DataFrame containing the spatial network with 'Cell1', 'Cell2', and 'Distance' columns.
            label (pd.Series): Series mapping cell IDs to their labels.

        Returns:
            pd.DataFrame: Pruned spatial network DataFrame.

        Example:
            >>> import pandas as pd
            >>> Graph_df = pd.DataFrame({'Cell1': ['cell1', 'cell2'], 'Cell2': ['cell2', 'cell3'], 'Distance': [1.0, 2.0]})
            >>> labels = pd.Series({'cell1': 'A', 'cell2': 'A', 'cell3': 'B'})
            >>> pruned_df = prune_spatial_Net(Graph_df, labels)
    """
    pro_labels_dict = dict(zip(label.index, label))
    Graph_df['Cell1_label'] = Graph_df['Cell1'].map(pro_labels_dict)
    Graph_df['Cell2_label'] = Graph_df['Cell2'].map(pro_labels_dict)
    Graph_df = Graph_df.loc[Graph_df['Cell1_label'] == Graph_df['Cell2_label'],]
    return Graph_df


def Cal_Precluster_Net(adata, is_pca=False, pcs=50, res_pre=0.6,verbose=False):
    """
        Constructs a pre-clustered spatial network based on expression data.

        Applies Louvain clustering to expression data (optionally after PCA) and prunes the spatial network to keep edges
        between cells in the same cluster.

        Args:
            adata (AnnData): AnnData object containing expression and spatial data.
            is_pca (bool, optional): If True, applies PCA before clustering, defaults to False.
            pcs (int, optional): Number of principal components for PCA, defaults to 50.
            res_pre (float, optional): Resolution for Louvain clustering, defaults to 0.6.
            verbose (bool, optional): If True, prints progress and statistics, defaults to False.

        Example:
            >>> import scanpy as sc
            >>> adata = sc.read_h5ad('data.h5ad')
            >>> Cal_Precluster_Net(adata, is_pca=True, pcs=30)
    """
    start = time.time()
    if verbose:
        tqdm.write('------Calculating precluster graph...')
    adata_var = adata.copy()
    if is_pca:
        if verbose:
            tqdm.write(f"begining pcs use n_com = {pcs}")
        if 'X_pca' not in adata.obsm.keys():
            sc.pp.pca(adata_var,n_comps=pcs,svd_solver='arpack')
        sc.pp.neighbors(adata_var,use_rep="X_pca")
        sc.tl.louvain(adata_var, resolution=res_pre, key_added='expression_louvain_label',random_state=42)
        pre_cluster = 'expression_louvain_label'
        prune_G_df = prune_spatial_Net(adata.uns['Spatial_Net'].copy(), adata_var.obs[pre_cluster])
        adata.uns['Precluster_Net'] = prune_G_df
        # adata.obs[pre_cluster] = adata_var.obs[pre_cluster].copy()
        if verbose:
            end = time.time()
            print(f'Precluster spatial graph contains {prune_G_df.shape[0]} edges, {adata.n_obs} cells.')  # Number of edges and cells
            print(
                f'{prune_G_df.shape[0] / adata.n_obs:.4f} neighbors per cell on average.')  # Average number of neighbors per cell
            print(f'Precluster graph calculation time: {end - start:.2f} seconds')

        del adata.uns['Precluster_Net']['Cell1_label']
        del adata.uns['Precluster_Net']['Cell2_label']
    else:
        sc.pp.neighbors(adata_var)
        sc.tl.louvain(adata_var, resolution=res_pre, key_added='expression_louvain_label',random_state=45)
        pre_cluster = 'expression_louvain_label'
        prune_G_df = prune_spatial_Net(adata.uns['Spatial_Net'].copy(), adata_var.obs[pre_cluster])
        adata.uns['Precluster_Net'] = prune_G_df
        # adata.obs[pre_cluster] = adata_var.obs[pre_cluster].copy()
        del adata.uns['Precluster_Net']['Cell1_label']
        del adata.uns['Precluster_Net']['Cell2_label']
        if verbose:
            end = time.time()
            print(
                f'Precluster spatial graph contains {prune_G_df.shape[0]} edges, {adata.n_obs} cells.')  # Number of edges and cells
            print(
                f'{prune_G_df.shape[0] / adata.n_obs:.4f} neighbors per cell on average.')  # Average number of neighbors per cell
            print(f'Precluster graph calculation time: {end - start:.2f} seconds')


def Transfer_Graph_Data(adata, dim_reduction=None, center_msg='out'):
    """
    Construct graph data for training.

    Parameters
    ----------
    adata : AnnData
        AnnData object which contains Spatial network and Expression network.
    dim_reduction : str or None
        Dimensional reduction methods (or the input feature). Can be 'PCA',
        'HVG' or None (default uses all gene expression, which may cause out of memory during training).
    center_msg : str
        Message passing mode through the graph. Given a center spot,
        'in' denotes that the message is flowing from connected spots to the center spot,
        'out' denotes that the message is flowing from the center spot to the connected spots.

    Returns
    -------
    data : Data
        The constructed graph data containing edges and features for training.
    """

    # Expression edge construction
    G_df = adata.uns['Precluster_Net'].copy()  # Copy the expression network DataFrame
    cells = np.array(adata.obs_names)  # Get cell names from AnnData object
    # Create a mapping from cell names to indices
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    # Map cell names to their corresponding indices
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    # Create a sparse matrix for the expression edges
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])  # Add self-loops to the graph
    exp_edge = np.nonzero(G)  # Get the non-zero indices of the expression graph

    # Spatial edge construction
    G_df = adata.uns['Spatial_Net'].copy()  # Copy the spatial network DataFrame
    cells = np.array(adata.obs_names)  # Get cell names from AnnData object
    # Create a mapping from cell names to indices
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    # Map cell names to their corresponding indices
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    # Create a sparse matrix for the spatial edges
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])  # Add self-loops to the graph
    spatial_edge = np.nonzero(G)  # Get the non-zero indices of the spatial graph

    # Feature extraction based on the specified dimensional reduction method
    if dim_reduction == 'PCA':
        feat = adata.obsm['X_pca']  # Use PCA-reduced features
    elif dim_reduction == 'HVG':
        # Filter AnnData for highly variable genes
        adata_Vars = adata[:, adata.var['highly_variable']]
        # Convert sparse matrix to dense array if necessary
        if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
            feat = adata_Vars.X.toarray()  # Convert to dense array
        else:
            feat = adata_Vars.X
    else:
        # Use all gene expression data if no dimensional reduction is specified
        if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
            feat = adata.X.toarray()  # Convert to dense array
        else:
            feat = adata.X  # Use the existing dense array

    # Construct the graph data based on the message passing mode
    if center_msg == 'out':
        # Create edge indices for outgoing messages

        data = Data(edge_index=torch.LongTensor(np.array(
            [np.concatenate((exp_edge[0], spatial_edge[0])),
             np.concatenate((exp_edge[1], spatial_edge[1]))])).contiguous(),
                    x=torch.FloatTensor(feat.copy()))  # Use feature tensor
    else:
        # Create edge indices for incoming messages
        data = Data(edge_index=torch.LongTensor(np.array(
            [np.concatenate((exp_edge[1], spatial_edge[1])),
             np.concatenate((exp_edge[0], spatial_edge[0]))])).contiguous(),
                    x=torch.FloatTensor(feat.copy()))

    # Create edge types for the constructed graph

    edge_type = torch.zeros(exp_edge[0].shape[0] + spatial_edge[0].shape[0], dtype=torch.int64)
    edge_type[exp_edge[0].shape[0]:] = 1  # Set spatial edges to type 1
    data.edge_type = edge_type  # Assign edge types to the data

    return data  # Return the constructed graph data


def Batch_Data(adata, num_batch_x, num_batch_y, plot_Stats=False):
    """
    Create batches of spatial data based on specified coordinates.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing spatial information.
    num_batch_x : int
        Number of batches along the x-axis.
    num_batch_y : int
        Number of batches along the y-axis.
    plot_Stats : bool, optional
        If True, plot statistics of the number of spots in each batch.

    Returns
    -------
    Batch_list : list
        A list containing AnnData objects for each batch.
    """

    # Retrieve spatial coordinates from the AnnData object
    Sp_df = adata.obsm['spatial']

    # Calculate the x-coordinates for the specified number of batches
    batch_x_coor = np.percentile(Sp_df[:, 0], np.linspace(0, 100, num_batch_x + 1))
    # Calculate the y-coordinates for the specified number of batches
    batch_y_coor = np.percentile(Sp_df[:, 1], np.linspace(0, 100, num_batch_y + 1))

    # Initialize an empty list to store each batch of data
    Batch_list = []

    # Iterate over the number of batches along the x-axis
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

            # Create a temporary AnnData object for the current batch based on the mask
            temp_adata = adata[mask].copy()
            # Check if the batch contains more than 10 spots
            if temp_adata.shape[0] > 10:
                Batch_list.append(temp_adata)  # Add the valid batch to the list

    # If plot_Stats is True, visualize the distribution of spots per batch
    if plot_Stats:
        f, ax = plt.subplots(figsize=(1, 3))  # Create a subplot for the boxplot
        # Create a DataFrame to hold the number of spots in each batch
        plot_df = pd.DataFrame([x.shape[0] for x in Batch_list], columns=['#spot/batch'])
        # Create a boxplot to show the distribution of spots per batch
        sns.boxplot(y='#spot/batch', data=plot_df, ax=ax)
        # Overlay a stripplot to show individual batch sizes
        sns.stripplot(y='#spot/batch', data=plot_df, ax=ax, color='red', size=5)

    return Batch_list  # Return the list of batches
