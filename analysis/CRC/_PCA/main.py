from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import os
import time
from tqdm import tqdm

# adata = sc.read_h5ad('/home/guo/jt/python/wave/big/crc/images/crc_clusterd.h5ad')
# print(adata.obsm['X_pca'].shape)

# adata = sc.read_visium('/home/guo/jt/data/CRC')
adata = sc.read_visium('/home/cavin/jt/spatial_data/bighumancrc')
adata.obs_names_make_unique()
adata.var_names_make_unique()
print(adata)

tqdm.write('------preprocesing data...')
start = time.time()
adata.X = adata.X.astype(np.float32)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000,subset=True)
# sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True)
# sc.pp.scale(adata,zero_center=True, max_value=10)
# adata = adata[:, adata.var.highly_variable]
tqdm.write(f'take times:{time.time()-start}')
tqdm.write('------running pca...')
start = time.time()
sc.pp.pca(adata, svd_solver='randomized', n_comps=50)
tqdm.write(f'take times:{time.time()-start}')
print(adata.obsm['X_pca'].shape)
sc.pp.neighbors(adata, use_rep='X_pca')
sc.tl.umap(adata)
# sc.tl.leiden(adata, random_state=2025, resolution=0.1)
sc.tl.louvain(adata, random_state=2025, resolution=0.05,key_added='leiden')
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
ax = sc.pl.embedding(adata, basis="spatial",show=False, s=0.3, color='leiden',title='spatial clustering result of PCA',cmap='tab20')
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
