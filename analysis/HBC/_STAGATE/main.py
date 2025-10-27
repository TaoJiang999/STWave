import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import time
# sys.path.append('/home/guo/jt/python/STAGATE_pyG')
# sys.path.append('/home/cavin/jt/python/STAGATE_pyG')


path_appended = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))+'/time_no_dic/_stagate'
print(path_appended)
sys.path.append(path_appended)

# sys.path.append('/home/guo/jt/python/STAGATE_pyG')
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score, homogeneity_score
import STAGATE_pyG

# os.environ['R_HOME'] = '/home/guo/anaconda3/envs/tao/lib/R'

# adata = sc.read_h5ad('/home/guo/jt/data/CRC/crc_cal_net.h5ad')
# adata = sc.read_h5ad('/home/cavin/jt/spatial_data/bighumancrc/crc_cal_net.h5ad')
# adata = sc.read_h5ad('/home/guo/jt/data/CRC/crc_cal_net.h5ad')
# adata = sc.read_visium('/home/cavin/jt/spatial_data/bighumancrc')
# adata = sc.read_h5ad('/home/guo/jt/data/bighumanbreast/bighumanbreast.h5ad')
adata = sc.read_h5ad('/home/waas/18161127346/data/big/bighumanbreast_filter_unknow_whith_cell_type.h5ad')

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
tqdm.write(f'take times:{time.time()-start}')
print("adata:", adata)


STAGATE_pyG.Cal_Spatial_Net(adata, k_cutoff=8, model='KNN')
adata = STAGATE_pyG.train_STAGATE(adata,n_epochs=1000,reduce='HVG')


n_clusters = adata.obs['cell_type'].nunique()
print('n_clusters:', n_clusters)
# sc.pp.neighbors(adata, use_rep='STAGATE')
# sc.tl.umap(adata)
adata = STAGATE_pyG.mclust_R(adata, used_obsm='STAGATE', num_cluster=n_clusters)

label = adata.obs['mclust']
dir = os.path.dirname(__file__)
label.to_csv(dir+'/pre_label.csv')
obs_df = adata.obs.dropna()
NMI_score = normalized_mutual_info_score(obs_df['mclust'], obs_df['cell_type'], average_method='max')
HS_score = homogeneity_score(obs_df['mclust'], obs_df['cell_type'])
adata.obs['domain'] = adata.obs['mclust'].copy()
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

data = {
    'Metric': ['ARI', 'NMI', 'HS'],
    'Score': [ARI_score, NMI_score, HS_score]
}

df = pd.DataFrame(data)

df.to_csv(dir+'/metric.csv', index=False)




# sc.pp.neighbors(adata, use_rep='STAGATE')
# sc.tl.umap(adata)
# # sc.tl.leiden(adata, random_state=2025, resolution=0.3)
# sc.tl.louvain(adata, random_state=2025, resolution=0.3,key_added='leiden')
# # adata = mclust_R(adata, used_obsm='DeepWave', num_cluster=cluster_num)

# ax = sc.pl.spatial(adata, basis="spatial",show=False, spot_size=5.0,color='leiden',title='spatial clustering result of STAGATE',cmap='Pastel1')
# # ax.invert_yaxis()
# plt.axis('off')
# save_path = os.path.dirname(__file__)+'/images/'
# os.makedirs(save_path, exist_ok=True)
# formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
# filename_time = formatted_time.replace(" ", "_").replace(":", "")
# plt.savefig(save_path+filename_time+'_crc.png',dpi=300, bbox_inches='tight')
# plt.close()
# print(f"cluster:{adata.obs['leiden'].nunique()}")
# df = adata.obs['leiden']
# df.to_csv(os.path.dirname(__file__)+'/images/'+filename_time+'_leiden.csv')
# # adata.write_h5ad(save_path+'crc_clusterd.h5ad')
print('finished')







