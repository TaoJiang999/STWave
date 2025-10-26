import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('/home/guo/jt/python/wave/time_no_dic/_spaceflow')
from SpaceFlow import SpaceFlow
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score, homogeneity_score
import os
# from myutils import measure_resources
import time
from tqdm import tqdm
import pandas as pd
# @measure_resources
def loading_and_preprocess_data():
    # adata = sc.read_h5ad('/home/guo/jt/data/simulate/simu1/rep1/data.h5ad')
    # adata = sc.read_h5ad('/gz-data/simulate/simu1/rep1/data.h5ad')
    # adata = sc.read_h5ad('/home/guo/jt/data/simulate/ad/data_sp_320k.h5ad')
    adata = sc.read_h5ad('/home/guo/jt/data/Embryo/E16.5_E1S3.MOSTA.h5ad')
    # adata = adata[:5000]
    tqdm.write('------preprocesing data...')
    start = time.time()
    adata.X = adata.X.astype(np.float32)
    # sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000,subset=True)
    # adata = adata[:, adata.var.highly_variable]
    # if sp.issparse(adata.X):
    #     count_matrix = adata.X.toarray()
    # else:
    #     count_matrix = np.array(adata.X)
    sf = SpaceFlow.SpaceFlow(adata=adata)
    # sf = SpaceFlow.SpaceFlow(adata=adata,count_matrix=count_matrix.all(), spatial_locs=adata.obsm['spatial'], sample_names=adata.obs_names, gene_names=adata.var_names)
    sf.preprocessing_data(n_top_genes=3000)
    tqdm.write(f'take times:{time.time()-start}')
    
    
    return adata,sf


# @measure_resources
def train(adata,sf:SpaceFlow.SpaceFlow):
    embedding = sf.train(spatial_regularization_strength=0.1, 
         z_dim=100, 
         lr=1e-3, 
         epochs=10, 
         max_patience=50, 
         min_stop=1000, 
         random_seed=42, 
         gpu=0, 
         regularization_acceleration=True, 
         edge_subset_sz=1000000)
    adata.obsm['emb'] = embedding
    sf.segmentation( 
                n_neighbors=50, 
                resolution=0.95)
    adata.obs['domain'] = np.array(sf.domains).astype(int)
    adata.obs['domain'] = adata.obs['domain'].astype(str)
    adata.obs['domain'] = adata.obs['domain'].astype('category')
    print('num cluster:',adata.obs['domain'].nunique())
    return adata,sf



# @measure_resources
def save_data(adata):
    label = adata.obs['domain']
    dir = os.path.dirname(__file__)
    label.to_csv(dir+'/pre_label_e1s3.csv')
    adata.obs['mclust'] = adata.obs['domain'].copy()
    obs_df = adata.obs.dropna()
    NMI_score = normalized_mutual_info_score(obs_df['mclust'], obs_df['annotation'], average_method='max')
    HS_score = homogeneity_score(obs_df['mclust'], obs_df['annotation'])
    adata.obs['domain'] = adata.obs['mclust'].copy()
    # new_type = refine_label(adata, radius=10, key='domain')
    # adata.obs['domain'] = new_type
    filtered_domain = adata.obs['domain'][obs_df.index]  # 按照obs_df的索引过滤domain
    filtered_ground_truth = obs_df['annotation']
    assert len(filtered_domain) == len(
        filtered_ground_truth), f"Shape mismatch: domain has {len(filtered_domain)} elements, ground_truth has {len(filtered_ground_truth)} elements"
    ARI_score = adjusted_rand_score(filtered_domain, filtered_ground_truth)
    print('ARI:', ARI_score)
    print('NMI:', NMI_score)
    print('HS:', HS_score)

    # 创建数据字典
    data = {
        'Metric': ['ARI', 'NMI', 'HS'],
        'Score': [ARI_score, NMI_score, HS_score]
    }
    # 创建 DataFrame
    df = pd.DataFrame(data)
    # 保存为 CSV 文件
    df.to_csv(dir+'/metric_e1s3.csv', index=False)

    
if __name__ == "__main__":
     adata,sf = loading_and_preprocess_data()
     adata,sf = train(adata,sf)
     save_data(adata)

    