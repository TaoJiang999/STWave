import warnings
warnings.filterwarnings('ignore')
import sys
import os
path_appended = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))+'/time_no_dic/_spaceflow'
print(path_appended)
sys.path.append(path_appended)
from SpaceFlow import SpaceFlow
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score, homogeneity_score
# from myutils import measure_resources
import time
from tqdm import tqdm
import pandas as pd
# @measure_resources
def loading_and_preprocess_data():
    # adata = sc.read_h5ad('/home/guo/jt/data/simulate/simu1/rep1/data.h5ad')
    # adata = sc.read_h5ad('/gz-data/simulate/simu1/rep1/data.h5ad')
    # adata = sc.read_h5ad('/home/guo/jt/data/simulate/ad/data_sp_320k.h5ad')
    # adata = sc.read_h5ad('/home/cavin/jt/spatial_data/big_lung_cancer/SMI_Lung.h5ad') /home/waas/18161127346/data/big/SMI_Lung.h5ad
    adata = sc.read_h5ad('/home/waas/18161127346/data/big/SMI_Lung.h5ad') 
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
    sf.preprocessing_data(n_top_genes=adata.shape[1])
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

    # for res in res_list:
    #     sf.segmentation( 
    #                 n_neighbors=50, 
    #                 resolution=0.5)
    #     adata.obs['domain'] = np.array(sf.domains).astype(int)
    #     adata.obs['domain'] = adata.obs['domain'].astype(str)
    return adata,sf



# @measure_resources
def save_data(adata):
    sc.pp.neighbors(adata, use_rep='emb')
    # sc.tl.umap(adata)

    res_list = [0.05,0.1,0.2,0.3,0.5,1,1.5,2]
    results = []
    cluster_results = []
    for res in res_list:
        sc.tl.leiden(adata, random_state=2024, resolution=res,key_added='domain')

        label = adata.obs['domain']
        dir = os.path.dirname(__file__)
        label.to_csv(dir+f'/res_{res}_cluster_{adata.obs['domain'].nunique()}_pre_label.csv')
        adata.obs['mclust'] = adata.obs['domain'].copy()
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
        results.append([res, ARI_score, NMI_score, HS_score])
        cluster_results.append(res,adata.obs['leiden'].nunique())
    df = pd.DataFrame(results, columns=['resolution', 'ARI', 'NMI', 'HS'])
  
    df.to_csv(dir+'/metric.csv', index=False)


    df = pd.DataFrame(cluster_results, columns=['resolution', 'n_cluster'])
 
    df.to_csv(dir+'/n_cluster.csv', index=False)


    
if __name__ == "__main__":
     adata,sf = loading_and_preprocess_data()
     adata,sf = train(adata,sf)
     save_data(adata)

    