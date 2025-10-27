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
adata = sc.read_h5ad('/home/cavin/jt/spatial_data/bighumanbreast/bighumanbreast_filter_unknow_whith_cell_type.h5ad')
adata.obs_names_make_unique()
adata.var_names_make_unique()
print(adata)

tqdm.write('------preprocesing data...')
start = time.time()
adata.X = adata.X.astype(np.float32)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=adata.shape[1],subset=True)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

adata = adata[:, adata.var.highly_variable]
tqdm.write(f'take times:{time.time()-start}')
tqdm.write('------running pca...')
start = time.time()
sc.pp.pca(adata, svd_solver='randomized', n_comps=50)
tqdm.write(f'take times:{time.time()-start}')
print(adata.obsm['X_pca'].shape)
sc.pp.neighbors(adata, use_rep='X_pca')
sc.tl.umap(adata)
# sc.tl.leiden(adata, random_state=2025, resolution=0.1)
sc.tl.louvain(adata, random_state=2025, resolution=1.9,key_added='leiden')
print(adata.obs['leiden'].unique())
# adata = mclust_R(adata, used_obsm='DeepWave', num_cluster=cluster_num)





label = adata.obs['leiden']
dir = os.path.dirname(__file__)
label.to_csv(dir+'/pre_label.csv')

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
print('ARI:', ARI_score)
print('NMI:', NMI_score)
print('HS:', HS_score)


data = {
    'Metric': ['ARI', 'NMI', 'HS'],
    'Score': [ARI_score, NMI_score, HS_score]
}

df = pd.DataFrame(data)

df.to_csv(dir+'/metric.csv', index=False)





# ax = sc.pl.spatial(adata, basis="spatial",show=False,spot_size=5, color='leiden',title='spatial clustering result of PCA',palette=rainbow_hex_10)
# # ax.invert_yaxis()
# plt.axis('off')
# save_path = os.path.dirname(__file__)+'/images/'
# os.makedirs(save_path, exist_ok=True)
# formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
# filename_time = formatted_time.replace(" ", "_").replace(":", "")
# plt.savefig(save_path+filename_time+'_hbc.png',dpi=300, bbox_inches='tight')
# plt.close()
# print(f"cluster:{adata.obs['leiden'].nunique()}")
# df = adata.obs['leiden']
# df.to_csv(os.path.dirname(__file__)+'/images/'+filename_time+'_leiden.csv')
# # adata.write_h5ad(save_path+'crc_clusterd.h5ad')
print('finished')
