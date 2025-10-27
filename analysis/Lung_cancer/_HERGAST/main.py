import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
import gzip
from scipy.io import mmread
import HERGAST
from sklearn.metrics.cluster import normalized_mutual_info_score, homogeneity_score
from sklearn.metrics import adjusted_rand_score





from myutils import measure_resources

# adata = sc.read_h5ad(f'/home/cavin/jt/spatial_data/big_lung_cancer/SMI_Lung.h5ad')
adata = sc.read_h5ad(f'/home/waas/18161127346/data/big/SMI_Lung.h5ad')
@measure_resources
def main(adata):
    data_path = 'data' #replace to your own path
    # adata = sc.read_h5ad(f'/home/cavin/jt/spatial_data/big_lung_cancer/SMI_Lung.h5ad')#/home/waas/18161127346/data/big/SMI_Lung.h5ad
    # adata = sc.read_h5ad(f'/home/waas/18161127346/data/big/SMI_Lung.h5ad')
    adata = adata[:5000]


    ###preprocess
    adata.raw = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True)
    sc.pp.scale(adata)
    # sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=adata.shape[1])
    # sc.pp.pca(adata, n_comps=200)
    adata.var['highly_variable'] = True
    HERGAST.utils.Cal_Spatial_Net(adata)
    HERGAST.utils.Cal_Expression_Net(adata, dim_reduce='HVG')

    train_HERGAST = HERGAST.Train_HERGAST(adata, batch_data=True, num_batch_x_y=(3,3), spatial_net_arg={'verbose':False},
                                        exp_net_arg={'verbose':False},dim_reduction='HVG',device_idx=0)

    train_HERGAST.train_HERGAST(n_epochs=200)
    return adata
    



adata = main(adata)

sc.pp.neighbors(adata, use_rep='HERGAST')

# sc.tl.umap(adata)

res_list = [0.05,0.1,0.2,0.3,0.5,1,1.5,2]
results = []
cluster_results = []
for res in res_list:

    sc.tl.leiden(adata, random_state=2024, resolution=res)

    label = adata.obs['leiden']
    dir = os.path.dirname(__file__)
    label.to_csv(dir+f'/res_{res}_cluster_{adata.obs.leiden.nunique()}_pre_label.csv')

    obs_df = adata.obs.dropna()
    label_index = 'cell_type'
    NMI_score = normalized_mutual_info_score(obs_df['leiden'], obs_df[label_index], average_method='max')
    HS_score = homogeneity_score(obs_df['leiden'], obs_df[label_index])
    adata.obs['domain'] = adata.obs['leiden'].copy()
    # new_type = refine_label(adata, radius=10, key='domain')
    # adata.obs['domain'] = new_type
    filtered_domain = adata.obs['domain'][obs_df.index]  # 按照obs_df的索引过滤domain
    filtered_ground_truth = obs_df[label_index]
    assert len(filtered_domain) == len(
            filtered_ground_truth), f"Shape mismatch: domain has {len(filtered_domain)} elements, ground_truth has {len(filtered_ground_truth)} elements"
    ARI_score = adjusted_rand_score(filtered_domain, filtered_ground_truth)
    print('current resolution:', res)
    print('ARI:', ARI_score)
    print('NMI:', NMI_score)
    print('HS:', HS_score)

    results.append([res, ARI_score, NMI_score, HS_score])
    cluster_results.append((res,adata.obs['leiden'].nunique()))


df = pd.DataFrame(results, columns=['resolution', 'ARI', 'NMI', 'HS'])

df.to_csv(dir+'/metric.csv', index=False)


df = pd.DataFrame(cluster_results, columns=['resolution', 'n_cluster'])

df.to_csv(dir+'/n_cluster.csv', index=False)