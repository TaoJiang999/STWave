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
sys.path.append('/home/cavin/jt/python/STAGATE_pyG')
# sys.path.append('/home/guo/jt/python/STAGATE_pyG')
from sklearn.metrics.cluster import adjusted_rand_score
import STAGATE_pyG

# os.environ['R_HOME'] = '/home/guo/anaconda3/envs/tao/lib/R'

# section_id = '151676'

# input_dir = os.path.join('/home/guo/jt/data/DLPFC', section_id)
# adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
# adata.var_names_make_unique()

# #Normalization
# sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
# STAGATE_pyG.Cal_Spatial_Net(adata, k_cutoff=8, model='KNN')

# adata = sc.read_h5ad('/home/guo/jt/data/CRC/crc_cal_net.h5ad')
# adata = sc.read_h5ad('/home/cavin/jt/spatial_data/bighumancrc/crc_cal_net.h5ad')
# adata = sc.read_h5ad('/home/guo/jt/data/CRC/crc_cal_net.h5ad')
adata = sc.read_visium('/home/cavin/jt/spatial_data/bighumancrc')
tqdm.write('------preprocesing data...')
start = time.time()
adata.X = adata.X.astype(np.float32)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000,subset=True)
sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True)
sc.pp.scale(adata,zero_center=True, max_value=10)
adata = adata[:, adata.var.highly_variable]
tqdm.write(f'take times:{time.time()-start}')
tqdm.write('------running pca...')
start = time.time()
sc.pp.pca(adata, svd_solver='randomized', n_comps=200)
tqdm.write(f'take times:{time.time()-start}')
print("adata:", adata)


STAGATE_pyG.Cal_Spatial_Net(adata, k_cutoff=6, model='KNN')
adata = STAGATE_pyG.train_STAGATE(adata,n_epochs=1000)

sc.pp.neighbors(adata, use_rep='STAGATE')
sc.tl.umap(adata)
# sc.tl.leiden(adata, random_state=2025, resolution=0.3)
sc.tl.louvain(adata, random_state=2025, resolution=0.3,key_added='leiden')
# adata = mclust_R(adata, used_obsm='DeepWave', num_cluster=cluster_num)
rainbow_hex = [
    '#FF6666',  # 鲜红
    '#FFB266',  # 橙黄
    '#FFFF99',  # 淡黄
    '#99FF99',  # 亮绿
    '#99FFFF',  # 青色
    '#99CCFF',  # 浅蓝
    '#C299FF',  # 紫色
    "#000000"  # 黑色
]
ax = sc.pl.embedding(adata, basis="spatial",show=False, s=0.6, color='leiden',title='spatial clustering result of STAGATE',cmap='tab20')
ax.invert_yaxis()
plt.axis('off')
save_path = os.path.dirname(__file__)+'/images/'
os.makedirs(save_path, exist_ok=True)
formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
filename_time = formatted_time.replace(" ", "_").replace(":", "")
plt.savefig(save_path+filename_time+'_crc.png',dpi=300, bbox_inches='tight')
plt.close()
print(f"cluster:{adata.obs['leiden'].nunique()}")
df = adata.obs['leiden']
df.to_csv(os.path.dirname(__file__)+'/images/'+filename_time+'_leiden.csv')
# adata.write_h5ad(save_path+'crc_clusterd.h5ad')
print('finished')







