from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import os
import time
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.cluster import normalized_mutual_info_score, homogeneity_score
from sklearn.metrics import adjusted_rand_score

# adata = sc.read_h5ad('/home/guo/jt/python/wave/big/crc/images/crc_clusterd.h5ad')
# print(adata.obsm['X_pca'].shape)

# adata = sc.read_visium('/home/guo/jt/data/CRC')
# adata = sc.read_visium('/home/cavin/jt/spatial_data/bighumancrc')
adata = sc.read_h5ad('/home/cavin/jt/spatial_data/big_lung_cancer/SMI_Lung.h5ad')
adata.obs_names_make_unique()
adata.var_names_make_unique()
print(adata)

tqdm.write('------preprocesing data...')
start = time.time()
adata.X = adata.X.astype(np.float32)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=adata.shape[1],subset=True)
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)

adata = adata[:, adata.var.highly_variable]
tqdm.write(f'take times:{time.time()-start}')
tqdm.write('------running pca...')
start = time.time()
sc.pp.pca(adata, svd_solver='randomized', n_comps=50)
tqdm.write(f'take times:{time.time()-start}')
print(adata.obsm['X_pca'].shape)
sc.pp.neighbors(adata, use_rep='X_pca')

res_list = [0.05,0.1,0.2,0.3,0.5,1,1.5,2]
results = []
cluster_results = []
for res in res_list:
    sc.tl.leiden(adata, random_state=2024, resolution=res,key_added='leiden')
    label = adata.obs['leiden']
    dir = os.path.dirname(__file__)
    label.to_csv(dir+f'/images/res_{res}_cluster_{adata.obs.leiden.nunique()}_pre_label.csv')

    obs_df = adata.obs.dropna()
    label_index = 'cell_type'
    NMI_score = normalized_mutual_info_score(obs_df['leiden'], obs_df[label_index], average_method='max')
    HS_score = homogeneity_score(obs_df['leiden'], obs_df[label_index])
    adata.obs['domain'] = adata.obs['leiden'].copy()
    # new_type = refine_label(adata, radius=10, key='domain')
    # adata.obs['domain'] = new_type
    filtered_domain = adata.obs['domain'][obs_df.index]  
    filtered_ground_truth = obs_df[label_index]
    assert len(filtered_domain) == len(
            filtered_ground_truth), f"Shape mismatch: domain has {len(filtered_domain)} elements, ground_truth has {len(filtered_ground_truth)} elements"
    ARI_score = adjusted_rand_score(filtered_domain, filtered_ground_truth)
    print('current resolution:', res)
    print('ARI:', ARI_score)
    print('NMI:', NMI_score)
    print('HS:', HS_score)

    results.append([res, ARI_score, NMI_score, HS_score])
    cluster_results.append([res,adata.obs['leiden'].nunique()])


df = pd.DataFrame(results, columns=['resolution', 'ARI', 'NMI', 'HS'])

df.to_csv(dir+'/images/metric.csv', index=False)


df = pd.DataFrame(cluster_results, columns=['resolution', 'n_cluster'])

df.to_csv(dir+'/images/n_cluster.csv', index=False)








