import torch
import scanpy as sc
import sys
import os
path_appended = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))+'/time_no_dic/_spaseg/SpaSEG'
print(path_appended)
sys.path.append(path_appended)
from os import path
from sklearn.metrics import adjusted_rand_score
import numpy as np
import seaborn as sns
import time
# from myutils import measure_resources
from sklearn.metrics.cluster import normalized_mutual_info_score, homogeneity_score
from tqdm import tqdm
# import sys
# sys.path.append(os.path.dirname(__file__)+"/SpaSEG/")
import spaseg
from spaseg import spaseg
from data_processing import scanpy_processing
from data_processing.scanpy_processing import sc_processing
import pandas as pd

# @measure_resources
def loading_and_preprocess_data():
    # adata = sc.read_h5ad('/home/guo/jt/data/simulate/simu1/rep1/data.h5ad')
    # adata = sc.read_h5ad('/gz-data/simulate/simu1/rep1/data.h5ad')
    # adata = sc.read_h5ad('/home/guo/jt/data/simulate/ad/data_40k.h5ad')
    # adata = sc.read_h5ad('/home/cavin/jt/spatial_data/big_lung_cancer/SMI_Lung.h5ad') #/home/waas/18161127346/data/big/SMI_Lung.h5ad
    adata = sc.read_h5ad('/home/waas/18161127346/data/big/SMI_Lung.h5ad') 
    n = np.ceil(np.sqrt(adata.shape[0]))
    adata.obs['array_col'] = adata.obsm['spatial'][:,0]/n
    adata.obs['array_row'] = adata.obsm['spatial'][:,1]/n
    # adata = adata[:5000]
    sample_id = 'simulate'
    multi_slice = False
    adata_list = [adata]
    sample_id_list = [sample_id]
    adata_list = sc_processing(adata_list=adata_list,
              sample_id_list=sample_id_list,
              multi_slice=multi_slice,
              st_platform="Visium",
              drop_cell_ratio=0.05,
              min_cells=5,
              compons=50)
    adata = adata_list[0]
    # tqdm.write('------preprocesing data...')
    # start = time.time()
    # adata.X = adata.X.astype(np.float32)
    # sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000,subset=True)
    # # sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True)
    # # sc.pp.scale(adata,zero_center=True, max_value=10)
    # adata = adata[:, adata.var.highly_variable]
    # tqdm.write(f'take times:{time.time()-start}')
    # tqdm.write('------running pca...')
    # start = time.time()
    # sc.pp.pca(adata, svd_solver='arpack', n_comps=200)
    # tqdm.write(f'take times:{time.time()-start}')
    # sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=200,subset=True)
    # adata = adata[:, adata.var.highly_variable]
    # adata.X = adata.obsm['X_pca'].copy()
    print("adata:", adata)
    return [adata]


# @measure_resources
def train(adata_list):

    alpha=0.4; beta=0.7
    torch.cuda.empty_cache()
    barcode_index="index"
    n_clusters = adata[0].obs['cell_type'].nunique()
    print('n_clusters:', n_clusters)
    # initilize SpaSEG model parameters
    spaseg_model = spaseg.SpaSEG(adata=adata_list,
                                use_gpu=True,
                                device="cuda:0",
                                input_dim=50,
                                nChannel=50,
                                output_dim=50,
                                sim_weight=alpha,
                                con_weight=beta,
                                min_label=n_clusters
                                )

    # prepare image-like tensor data for SpaSEG model input
    input_mxt, H, W = spaseg_model._prepare_data()

    # SpaSEG traning
    cluster_label, embedding = spaseg_model._train(input_mxt)
    print('embedding shape:', embedding.shape)
    embedding = embedding.reshape(-1, int(np.ceil(np.sqrt(embedding.shape[1]))),embedding.shape[-1])
    embedding = embedding[adata[0].obs['array_col'].values.astype('int'), adata[0].obs['array_row'].values.astype('int'), :]
    print('reshaped embedding shape:', embedding.shape)
    adata[0].obsm['emb'] = embedding
    # n_batch = 1
    # spaseg_model._add_seg_label(cluster_label, n_batch, H, W, barcode_index="unique_ID")
    print(adata)
    # adata_list[0].obsm['emb'] = embedding
    # adata_list[0].obs['mclust'] = embedding
    # adata_list[0].obs['mclust'] = adata_list[0].obs['mclust'].astype('int')
    # adata_list[0].obs['mclust'] = adata_list[0].obs['mclust'].astype('category')
    return adata_list


# @measure_resources
def save_data(adata):
    sc.pp.neighbors(adata, use_rep='emb')
    res_list = [0.05,0.1,0.2,0.3,0.5,1,1.5,2]
    results = []
    cluster_results = []
    for res in res_list:
        sc.tl.leiden(adata, random_state=2024, resolution=res,key_added='SpaSEG_discrete_clusters')

        label = adata.obs['SpaSEG_discrete_clusters']
        dir = os.path.dirname(__file__)
        label.to_csv(dir+f'/res_{res}_cluster_{adata.obs['SpaSEG_discrete_clusters'].nunique()}_pre_label.csv')
        adata.obs['mclust'] = adata.obs['SpaSEG_discrete_clusters'].copy()
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
     adata = loading_and_preprocess_data()
     train(adata)
     save_data(adata[0])
