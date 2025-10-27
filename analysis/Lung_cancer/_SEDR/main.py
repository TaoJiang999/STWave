import sys
import os
path_appended = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))+'/time_no_dic/_sedr'
print(path_appended)
sys.path.append(path_appended)
import scanpy as sc
import pandas as pd
from sklearn import metrics
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances, adjusted_rand_score, normalized_mutual_info_score, silhouette_score, davies_bouldin_score
from sklearn.metrics.cluster import homogeneity_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import SEDR
from tqdm import tqdm
print('import finished')
random_seed = 2023
SEDR.fix_seed(random_seed)
import time
# from myutils import measure_resources
# gpu
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# path

# @measure_resources
def loading_and_preprocess_data():
    # adata = sc.read_h5ad('/home/guo/jt/data/simulate/simu1/rep1/data.h5ad')
    # adata = sc.read_h5ad('/gz-data/simulate/simu1/rep1/data.h5ad')
    # adata = sc.read_h5ad('/home/cavin/jt/spatial_data/big_lung_cancer/SMI_Lung.h5ad') /home/waas/18161127346/data/big/SMI_Lung.h5ad
    adata = sc.read_h5ad('/home/waas/18161127346/data/big/SMI_Lung.h5ad')   
    adata.var_names_make_unique()
    # adata = adata[:5000]

    tqdm.write('------preprocesing data...')
    start = time.time()
    adata.X = adata.X.astype(np.float32)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=adata.shape[1],subset=True)
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

# @measure_resources
def train(adata):
    dim_reduction = 'HVG'
    if dim_reduction == 'PCA':
            if 'X_pca' not in adata.obsm.keys():
                raise ValueError("PCA has not been done! Run sc.pp.pca first!")


    graph_dict = SEDR.graph_construction(adata, n=8)
    print(graph_dict)

    sedr_net = SEDR.Sedr(adata.X.copy(), graph_dict, mode='clustering', device=device)
    using_dec = True
    if using_dec:
        sedr_net.train_with_dec(N=1)
    else:
        sedr_net.train_without_dec(N=1)
    sedr_feat, _, _, _ = sedr_net.process()
    adata.obsm['SEDR'] = sedr_feat

# @measure_resources
def clustering(adata):
     print("------clustering...")
    # n_clusters = adata.obs['cell_type'].nunique()
    # print('n_clusters:', n_clusters)
    # SEDR.mclust_R(adata, n_clusters, use_rep='SEDR', key_added='SEDR')

# @measure_resources
def save_data(adata):

    res_list = [0.05,0.1,0.2,0.3,0.5,1,1.5,2]
    results = []
    cluster_results = []
    sc.pp.neighbors(adata, use_rep='SEDR')
    for res in res_list:
        sc.tl.leiden(adata, random_state=2024, resolution=res,key_added='SEDR')
        label_df = adata.obs['SEDR']
        dir = os.path.dirname(__file__)
        label_df.to_csv(dir+f'/res_{res}_cluster_{adata.obs['SEDR'].nunique()}_pre_label.csv')

        obs_df = adata.obs.dropna()
        NMI_score = normalized_mutual_info_score(obs_df['SEDR'], obs_df['cell_type'], average_method='max')
        HS_score = homogeneity_score(obs_df['SEDR'], obs_df['cell_type'])
        adata.obs['domain'] = adata.obs['SEDR'].copy()
        # new_type = refine_label(adata, radius=10, key='domain')
        # adata.obs['domain'] = new_type
        filtered_domain = adata.obs['domain'][obs_df.index]  
        filtered_ground_truth = obs_df['cell_type']
        assert len(filtered_domain) == len(
            filtered_ground_truth), f"Shape mismatch: domain has {len(filtered_domain)} elements, ground_truth has {len(filtered_ground_truth)} elements"
        ARI_score = adjusted_rand_score(filtered_domain, filtered_ground_truth)
        print('ARI:', ARI_score)
        print('NMI:', NMI_score)
        print('HS:', HS_score)
        results.append([res, ARI_score, NMI_score, HS_score])
        cluster_results.append(res,adata.obs['leiden'].nunique())
    df = pd.DataFrame(results, columns=['resolution', 'ARI', 'NMI', 'HS'])

    df.to_csv(dir+'/metric.csv', index=False)


    df = pd.DataFrame(cluster_results, columns=['resolution', 'n_cluster'])

    df.to_csv(dir+'/n_cluster.csv', index=False)


if __name__ == "__main__":
     adata = loading_and_preprocess_data()
     train(adata)
     clustering(adata)
     save_data(adata)

