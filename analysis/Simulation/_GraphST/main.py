import os
import torch
import pandas as pd
import scanpy as sc
from GraphST import GraphST
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score,homogeneity_score
import warnings
from tqdm import tqdm
import time
import numpy as np
# 忽略所有警告
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from myutils import measure_resources

@measure_resources
def loading_and_preprocess_data():
    # adata = sc.read_h5ad('/home/guo/jt/data/simulate/simu1/rep1/data.h5ad')
    adata = sc.read_h5ad('/gz-data/simulate/simu1/rep1/data.h5ad')
    # adata = adata[:5000]

    tqdm.write('------preprocesing data...')
    start = time.time()
    adata.X = adata.X.astype(np.float32)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000,subset=True)
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


@measure_resources
def train(adata):
    dim_reduction = 'HVG'
    if dim_reduction == 'PCA':
            if 'X_pca' not in adata.obsm.keys():
                raise ValueError("PCA has not been done! Run sc.pp.pca first!")
    elif dim_reduction == 'HVG':
            if 'highly_variable' not in adata.var.keys():
                raise ValueError("HVG has not been computed! Run sc.pp.highly_variable_genes first!")
    model = GraphST.GraphST(adata, device=device, random_seed=2025,dim_reduce='HVG')
    adata = model.train()
    return adata


@measure_resources
def clustering(adata):
    n_clusters = 9
    radius = 50
    tool = 'mclust'  # mclust, leiden, and louvain
    # clustering
    from GraphST.utils import clustering

    if tool == 'mclust':
        clustering(adata, n_clusters, radius=radius, method=tool,
                    refinement=False)  # For DLPFC dataset, we use optional refinement step.
    elif tool in ['leiden', 'louvain']:
        clustering(adata, n_clusters, radius=radius, method=tool, start=0.3, end=3, increment=0.02, refinement=False)


@measure_resources
def save_data(adata):
    label = adata.obs['mclust']
    dir = os.path.dirname(__file__)
    label.to_csv(dir+'/pre_label.csv')

    obs_df = adata.obs.dropna()
    NMI_score = normalized_mutual_info_score(obs_df['mclust'], obs_df['ann_level_3'], average_method='max')
    HS_score = homogeneity_score(obs_df['mclust'], obs_df['ann_level_3'])
    adata.obs['domain'] = adata.obs['mclust'].copy()
    filtered_domain = adata.obs['domain'][obs_df.index]  # 按照obs_df的索引过滤domain
    filtered_ground_truth = obs_df['ann_level_3']
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
    df.to_csv(dir+'/metric.csv', index=False)

if __name__ == "__main__":
     adata = loading_and_preprocess_data()
     adata = train(adata)
     clustering(adata)
     save_data(adata)

