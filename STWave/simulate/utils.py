import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import seaborn as sns
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
import random
from scipy.stats import nbinom
from skimage import io, color, transform
import warnings



def cal_metagene(adata, gene_list, obs_name='metagene', layer=None, normalize=True):
    """
    Calculate the metagene expression for a specified list of genes.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing gene expression data.
    gene_list : list of str
        List of gene names for which to calculate the metagene.
    obs_name : str
        Name of the observation to store the metagene expression in adata.obs.
    layer : str or None
        Optional layer from which to extract gene expressions. If None, uses the main expression matrix.

    Returns
    -------
    None
        The metagene expression is saved in adata.obs under the specified obs_name.
    """

    # Check if a specific layer is provided for gene expressions
    if layer is not None:
        # Extract gene expressions from the specified layer
        gene_expressions = adata[:, gene_list].layers[layer]
    else:
        # Extract gene expressions from the main expression matrix
        gene_expressions = adata[:, gene_list].X

    # Check if the gene expressions are in sparse format
    if sp.issparse(gene_expressions):
        # Convert sparse matrix to a dense array for easier manipulation
        gene_expressions = gene_expressions.toarray()

    if normalize:
        gene_expressions = _min_max_norm(gene_expressions)
        
    # Calculate the metagene expression by summing the expressions of the specified genes across cells
    metagene_expression = np.sum(gene_expressions, axis=1)

    # Store the calculated metagene expression in the AnnData object's observations
    adata.obs[obs_name] = metagene_expression





def simulate_ST(sc_adata, spatial_df, sc_type_col='ann_level_3', sp_type_col='domain', disperse_frac=0.3):
    """
    Simulate spatial transcriptomics data from single-cell RNA-seq data.
    
    Parameters:
    - sc_adata: AnnData object containing single-cell RNA-seq data.
    - spatial_df: DataFrame containing spatial coordinates and cell type labels.
    
    Returns:
    - AnnData object with spatial transcriptomics data. Spatial coordinates are in obsm['spatial'] and cell types are in obs.
    """
    
    # Get unique cell types from single-cell data
    unique_sc_types = sc_adata.obs[sc_type_col].unique()
    
    # Get unique cell types from spatial data
    unique_spatial_types = spatial_df[sp_type_col].unique()
    
    # Randomly select cell types for each spatial type
    selected_types = np.random.choice(unique_sc_types, len(unique_spatial_types), replace=False)
    
    # Create a mapping from spatial types to selected single-cell types
    type_mapping = dict(zip(unique_spatial_types, selected_types))
    
    # Initialize a list to collect the simulated cell indices
    simulated_indices = []
    spatial_coords = []
    
    # Iterate over each cell type in the spatial data
    for spatial_type, count in spatial_df[sp_type_col].value_counts().items():
        # Get the corresponding single-cell type
        sc_type = type_mapping[spatial_type]
        print(f'sptial type:{spatial_type}, sc type:{sc_type}')
        
        # Get all cells of this type from the single-cell data
        sc_cells_of_type_indices = np.where(sc_adata.obs[sc_type_col] == sc_type)[0]
        
        if len(sc_cells_of_type_indices) >= count:
            # If we have enough cells, randomly select 'count' cells
            selected_indices = np.random.choice(sc_cells_of_type_indices, count, replace=False)
        else:
            # If not enough cells, randomly assign the available cells to the spatial positions
            selected_indices = np.random.choice(sc_cells_of_type_indices, count, replace=True)

        # Collect the selected cell indices
        simulated_indices.extend(selected_indices)
        
        # Collect the corresponding spatial coordinates
        spatial_coords.append(spatial_df[spatial_df[sp_type_col] == spatial_type][['x', 'y']].values)

    # Randomly select a cell type to simulate the dispersed cells
    remaining_types = list(set(unique_sc_types) - set(selected_types))
    if remaining_types:
        dispersed_type = np.random.choice(remaining_types)
        print(f'disperse cell type:{dispersed_type}')
        dispersed_cells_indices = np.where(sc_adata.obs[sc_type_col] == dispersed_type)[0]
        dispersed_sample_size = round(spatial_df.shape[0] * disperse_frac)
        if dispersed_cells_indices.shape[0] >= dispersed_sample_size:
            dispersed_cells_indices = np.random.choice(dispersed_cells_indices, dispersed_sample_size, replace=False)
        else:
            dispersed_cells_indices = np.random.choice(dispersed_cells_indices, dispersed_sample_size, replace=True)
        
        # Replace some cells in the simulated_adata with dispersed cells
        replace_indices = np.random.choice(len(simulated_indices), dispersed_sample_size, replace=False)
        simulated_indices_array = np.array(simulated_indices)
        simulated_indices_array[replace_indices] = dispersed_cells_indices

    # Concatenate all selected cells to form the simulated spatial data
    simulated_adata = sc_adata[simulated_indices_array].copy()
    simulated_adata.obs_names_make_unique()

    # Set the spatial coordinates
    simulated_adata.obsm['spatial'] = np.vstack(spatial_coords)
    simulated_adata.obs[sc_type_col] = simulated_adata.obs[sc_type_col].astype('str')
    simulated_adata.obs[sc_type_col].iloc[replace_indices] = dispersed_type
    
    return simulated_adata


def simulate_gene(lambda_val=0.7, spots=10000, se=50, ns=50, type='ZINB', se_p=0.3, 
            se_size=10, se_mu=10, ns_p=0.3, ns_size=5, ns_mu=5, ptn='2_ring.png',
            png_dir='/home/gongyuqiao/ur_annotation/Mytrain/amplification/ptn_png', outlier=False):
    
    """
    Simulate gene expression data based on a Poisson point process using image's pattern.

    Parameters:
    - lambda_val: Average number of points per unit area.
    - spots: Total number of points to simulate.
    - se: The number of  spatially variable genes (SVGs).
    - ns: The number of  non-SVGs.
    - type: Type of distribution for expression simulation ('ZINB' or 'ZIP').
    - se_p: Probability of zero-inflated expression for SVGs in the streak area.
    - se_size: Size parameter for zero-inflated distribution for SVGs in the streak area.
    - se_mu: For SVGs, the lambda para in the poisson distribution or the mu para in the NB distribution.
    - ns_p: Probability of zero-inflated expression of non-SVGs and SVGs in the non-streak area.
    - ns_size: For non-SVGs and SVGs in the non-streak area, the size para in the NB distribution.
    - ns_mu: For non-SVGs and SVGs in the non-streak area, the lambda para in the poisson
             distribution or the mu para in the NB distribution.
    - ptn: The file name of the pattern png image.
    - png_dir: Directory where the image is located.
    - outlier: Whether to simulate outliers in the expression data.

    Returns:
    - adata: AnnData object containing the simulated gene expression data and spatial coordinates.
    """

    win_size = int(np.ceil(np.sqrt(spots / lambda_val)))
    win = [0, win_size, 0, win_size]
    coor_x = _rpoispp(lambda_val, win)
    coor_dt = pd.DataFrame({
    'row': coor_x[:,0].astype(int),
    'col': coor_x[:,1].astype(int)
    })
    coor_dt = coor_dt.drop_duplicates().reset_index(drop=True)
    coor_dt['cell'] = ['c_'+str(i) for i in range(coor_dt.shape[0])]

    # Load and process the image
    image_path = f"{png_dir}/{ptn}"
    image = io.imread(image_path)
    gray_image = color.rgb2gray(image)
    re_img = transform.resize(gray_image, (win_size, win_size))

    # Convert the image into a binary mask
    img_coor = np.round(re_img)
    img_coords = np.argwhere(img_coor > 0)

    # Merge coordinates
    coor_s1 = coor_dt.merge(pd.DataFrame(img_coords, columns=['row', 'col']), on=['row', 'col'])

    # Extract marked and random coordinates
    coor_mark = coor_s1
    coor_random = coor_dt[~coor_dt.cell.isin(coor_mark.cell)]

    # Simulate expression for marked coordinates
    exp_mark = np.array([_simu_zi(family=type, subject_n=len(coor_mark), zi_p=se_p, size=se_size, mu=se_mu) for _ in range(se)]).T
    
    # Simulate expression for random coordinates
    exp_random = np.array([_simu_zi(family=type, subject_n=len(coor_random), zi_p=ns_p, size=ns_size, mu=ns_mu) for _ in range(se)]).T
    
    # Combine expression data
    exp_svg = np.vstack((exp_mark, exp_random))
    non_coor = pd.concat([coor_mark, coor_random])
    
    # Simulate non-SVG expression data
    exp_non = np.array([_simu_zi(family=type, subject_n=len(non_coor), zi_p=ns_p, size=ns_size, mu=ns_mu) for _ in range(ns)]).T
    
    # Combine all data
    all_data = np.hstack((non_coor[['row', 'col']], exp_svg, exp_non))
    
    # Handle outliers
    if outlier:
        if outlier < 0 or outlier >= 1:
            print("# outlier parameter is wrong!")
            end = all_data
        else:
            ind = random.sample(range(len(all_data)), round(len(all_data) * outlier))
            out_para = 5
            for idx in ind:
                all_data[idx, 2:] = _simu_zi(family=type, subject_n=(len(ind) * (all_data.shape[1] - 2)), 
                                              zi_p=se_p / 2, size=se_size * out_para, mu=se_mu * out_para)
            end = all_data
    else:
        end = all_data
    
    # Create a DataFrame for the results
    columns = ['row', 'col'] + [f'se.{i+1}' for i in range(se)] + [f'ns.{i+1}' for i in range(ns)]
    result_df = pd.DataFrame(end, columns=columns)
    adata = sc.AnnData(X=result_df.iloc[:,2:])
    adata.obsm['spatial'] = result_df.iloc[:,:2].to_numpy()
    adata.obs['mark_area'] = ['1']*coor_mark.shape[0] + ['0']*coor_random.shape[0]
    return adata


# internal helping functions

def _rpoispp(lambda_val, win):
    

    area = (win[1] - win[0]) * (win[3] - win[2])
    

    expected_points = np.random.poisson(lambda_val * area)
    

    x_coords = np.random.uniform(win[0], win[1], expected_points)
    y_coords = np.random.uniform(win[2], win[3], expected_points)
    
    return np.column_stack((x_coords, y_coords))

def _simu_zi(family, subject_n, zi_p=0.5, mu=0.5, size=0.25):
    Y = np.empty(subject_n)
    ind_mix = np.random.binomial(1, zi_p, size=subject_n)
    
    if family == "ZIP":
        Y[ind_mix != 0] = 0
        Y[ind_mix == 0] = np.random.poisson(mu, size=np.sum(ind_mix == 0))
    elif family == "ZINB":
        Y[ind_mix != 0] = 0
        Y[ind_mix == 0] = nbinom.rvs(n=size, p=1 / (1 + mu / size), size=np.sum(ind_mix == 0))
    
    return Y

def _min_max_norm(data):

    min_vals = data.min(axis=0)  
    max_vals = data.max(axis=0)  

    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data