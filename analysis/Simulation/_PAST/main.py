import past
import os
import scanpy as sc
import warnings
import torch
import numpy as np
import anndata as ad
import pandas as pd
from myutils import measure_resources
import time
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score,homogeneity_score
from tqdm import tqdm

# @measure_resources
def loading_and_preprocess_data():
    # adata = sc.read_h5ad('/home/guo/jt/data/simulate/simu1/rep1/data.h5ad')
    adata = sc.read_h5ad('/home/guo/jt/data/simulate/ad/data_640k.h5ad')
    # adata = adata[:5000]

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

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# @measure_resources
def train(adata):
    dim_reduction = 'PCA'
    if dim_reduction == 'PCA':
            if 'X_pca' not in adata.obsm.keys():
                raise ValueError("PCA has not been done! Run sc.pp.pca first!")
    elif dim_reduction == 'HVG':
            if 'highly_variable' not in adata.var.keys():
                raise ValueError("HVG has not been computed! Run sc.pp.highly_variable_genes first!")
    past.setup_seed(666)
    # sdata = past.preprocess(sdata, min_cells=3)
    # X_pca = adata.obsm['X_pca'].copy()  # shape = [5000, 50]
    # obs = adata.obs.copy()
    # var = pd.DataFrame(index=[f'PC{i+1}' for i in range(X_pca.shape[1])])  # shape = [50, 0]

    # adata_pca = ad.AnnData(X=X_pca, obs=obs, var=var)
    # adata_pca.obsm = adata.obsm.copy()
    # adata_raw = adata.copy()
    # adata = adata_pca
    # del adata_pca
    PAST_model = past.PAST(d_in=adata.X.shape[1], d_lat=30, k_neighbors=8, dropout=0.1).to(device)
    PAST_model.model_train(adata, epochs=50, lr=1e-3, device=device)
    adata = PAST_model.output(adata)
    return adata

# @measure_resources
def clustering(adata):
    # adata = past.mclust_R(adata, num_cluster=adata.obs['ann_level_3'].nunique(), used_obsm='embedding')
    adata = past.default_louvain(adata, use_rep="embedding")

# @measure_resources
def save_data(adata):
    label = adata.obs['louvain']
    dir = os.path.dirname(__file__)
    label.to_csv(dir+'/pre_label_2.csv')

    obs_df = adata.obs.dropna()
    NMI_score = normalized_mutual_info_score(obs_df['louvain'], obs_df['ann_level_3'], average_method='max')
    HS_score = homogeneity_score(obs_df['louvain'], obs_df['ann_level_3'])
    adata.obs['domain'] = adata.obs['louvain'].copy()
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

    df.to_csv(dir+'/metric_2.csv', index=False)

if __name__ == "__main__":
     adata = loading_and_preprocess_data()
     adata = train(adata)
     clustering(adata)
     save_data(adata)