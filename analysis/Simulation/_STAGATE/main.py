import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score, homogeneity_score
import STAGATE_pyG
from tqdm import tqdm
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
import time
from myutils import measure_resources
# the location of R (used for the mclust clustering)
# os.environ['R_HOME'] = 'D:\R-4.4.2'
# os.environ['R_USER'] = 'D:\ProgramData\Anaconda3\Lib\site-packages\rpy2'


@measure_resources
def loading_and_preprocess_data():
    # adata = sc.read_h5ad('/home/guo/jt/data/simulate/simu1/rep1/data.h5ad')
    # adata = sc.read_h5ad('/gz-data/simulate/simu1/rep1/data.h5ad')
    adata = sc.read_h5ad('/home/guo/jt/data/simulate/ad/data_5k.h5ad')
    # adata = adata[:5000]
    STAGATE_pyG.Cal_Spatial_Net(adata,k_cutoff=8,model='KNN')

    tqdm.write('------preprocesing data...')
    start = time.time()
    adata.X = adata.X.astype(np.float32)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000,subset=True)
    # sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True)
    # sc.pp.scale(adata,zero_center=True, max_value=10)
    adata = adata[:, adata.var.highly_variable]
    tqdm.write(f'take times:{time.time()-start}')
    # tqdm.write('------running pca...')
    # start = time.time()
    # sc.pp.pca(adata, svd_solver='arpack', n_comps=200)
    # tqdm.write(f'take times:{time.time()-start}')
    print("adata:", adata)
    return adata
@measure_resources
def train(adata):
    dim_reduction = 'HVG'
    if dim_reduction == 'PCA':
            if 'X_pca' not in adata.obsm.keys():
                raise ValueError("PCA has not been done! Run sc.pp.pca first!")
    elif dim_reduction == 'HVG':
            if 'highly_variable' not in adata.var.keys():
                raise ValueError("HVG has not been computed! Run sc.pp.highly_variable_genes first!")
            
    if 'Spatial_Net' not in adata.uns.keys():
                raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")


    spatial_net_arg = {'k_cutoff':8, 'model':'KNN', 'verbose':False}
    adata = STAGATE_pyG.train_STAGATE(adata,reduce=dim_reduction)
    print('adata:',adata)

@measure_resources
def clustering(adata):

    num_cluster = 9
    # sc.pp.neighbors(adata, use_rep='STAGATE')
    # sc.tl.umap(adata)
    adata = STAGATE_pyG.mclust_R(adata, used_obsm='STAGATE', num_cluster=num_cluster)

@measure_resources
def save_data(adata):
    label = adata.obs['mclust']
    dir = os.path.dirname(__file__)
    label.to_csv(dir+'/pre_label.csv')

    obs_df = adata.obs.dropna()
    NMI_score = normalized_mutual_info_score(obs_df['mclust'], obs_df['ann_level_3'], average_method='max')
    HS_score = homogeneity_score(obs_df['mclust'], obs_df['ann_level_3'])
    adata.obs['domain'] = adata.obs['mclust'].copy()
    # new_type = refine_label(adata, radius=10, key='domain')
    # adata.obs['domain'] = new_type
    filtered_domain = adata.obs['domain'][obs_df.index]  
    filtered_ground_truth = obs_df['ann_level_3']
    assert len(filtered_domain) == len(
        filtered_ground_truth), f"Shape mismatch: domain has {len(filtered_domain)} elements, ground_truth has {len(filtered_ground_truth)} elements"
    ARI_score = adjusted_rand_score(filtered_domain, filtered_ground_truth)
    print('ARI:', ARI_score)
    print('NMI:', NMI_score)
    print('HS:', HS_score)


    data = {
        'Metric': ['ARI', 'NMI', 'HS'],
        'Score': [ARI_score, NMI_score, HS_score]
    }

    df = pd.DataFrame(data)

    df.to_csv(dir+'/metric.csv', index=False)

if __name__ == "__main__":
     adata = loading_and_preprocess_data()
     train(adata)
     clustering(adata)
     save_data(adata)

