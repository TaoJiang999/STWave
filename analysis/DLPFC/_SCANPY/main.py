# Core scverse libraries
import scanpy as sc
import anndata as ad
import os
# 获取当前工作目录
current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")
# Data retrieval
import pooch
import time

from matplotlib import pyplot as plt
from _utils import *
import torch
import torch.nn.functional as F
import torch_sparse
import scanpy as sc
import scipy.sparse as sp

import warnings
from tqdm import tqdm
from torch_geometric.data import Data
from typing import Optional,Union
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score, homogeneity_score
#os.environ['R_HOME'] = 'D:\\R-4.4.2'
warnings.filterwarnings('ignore')
device = select_device()
seed_everything(2025)

result = {}

def train(name):
     # Read data
    path = '/home/guo/jiangtao/data/DLPFC/'
    path = '/home/cavin/jt/spatial_data/stDCL/DLPFC/'
    adata = sc.read_visium(path + name, count_file='filtered_feature_bc_matrix.h5')
    print("adata:", adata)
    import pandas as pd
    ground_truth_df = pd.read_csv(path + name + '_truth.txt', delimiter='\t', header=None, dtype=str)
    adata.obs['ground_truth'] = ground_truth_df.iloc[:, 1].values  # 假设标签在第一列
    # Data preprocessing
    adata.var_names_make_unique()
    prefilter_genes(adata, min_cells=3)
    #adata, adata_raw = svg(adata, svg_method='gft_top', n_top=3000)
    adata = svg(adata, svg_method='seurat_v3', n_top=3000)#seurat_v3
    # Build spotnet and genenet
    obtain_spotnet(adata, knn_method='Radius', rad_cutoff=150)
    #obtain_pre_spotnet(adata, adata_raw)
    obtain_pre_spotnet(adata, adata)
    print('adata',adata)
    # print('adata_raw',adata_raw)
    print('adata_raw', adata)
    end = time.time()
    adata = adata[~adata.obs['ground_truth'].isnull()]
    cluster_num = 5 if name in ['151669', '151670', '151671', '151672'] else 7
    sc.tl.pca(adata, n_comps=50)
    sc.pp.neighbors(adata,use_rep='X_pca')
    rainbow_hex = [
    '#FF6666',  # 鲜红
    '#FFB266',  # 橙黄
    '#FFFF99',  # 淡黄
    '#99FF99',  # 亮绿
    '#99FFFF',  # 青色
    '#99CCFF',  # 浅蓝
    '#C299FF',  # 紫色
    '#D32F2F',  # 深红
    '#F57C00',  # 深橙
    '#FBC02D',  # 深黄
    '#388E3C',  # 深绿
    '#0097A7',  # 深青

    ]
    sc.tl.leiden(adata,resolution=1.0)
    plt.rcParams.update({'axes.titlesize':20})
    adata.obs['ground_truth'] = adata.obs['ground_truth'].astype(str)
    adata.obs['leiden'] = adata.obs['leiden'].astype(str)
    obs_df = adata.obs.dropna()
    NMI_score = normalized_mutual_info_score(obs_df['leiden'], obs_df['ground_truth'], average_method='max')
    HS_score = homogeneity_score(obs_df['leiden'], obs_df['ground_truth'])
    adata.obs['domain'] = adata.obs['leiden'].copy()
    filtered_domain = adata.obs['domain'][obs_df.index]  
    filtered_ground_truth = obs_df['ground_truth']

 
    assert len(filtered_domain) == len(filtered_ground_truth), f"Shape mismatch: domain has {len(filtered_domain)} elements, ground_truth has {len(filtered_ground_truth)} elements"

    ARI_score = adjusted_rand_score(filtered_domain, filtered_ground_truth)
    print("current slice:", name)
    print('ARI:', ARI_score)
    print('NMI:', NMI_score)
    print('HS:', HS_score)
    r = {'ARI':ARI_score,'NMI':NMI_score,'HS':HS_score}
    result[name] = r
    sc.pl.spatial(adata, color=["ground_truth", "domain"], palette=rainbow_hex[:cluster_num], title=['ground truth', 'Scanpy(ARI=%.2f)' % ARI_score],show=False)
    dir = os.path.dirname(__file__)
    os.makedirs(os.path.join(dir, 'images'), exist_ok=True)
    plt.savefig(dir+'/images/' + name + '.svg', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot UMAP
    sc.tl.umap(adata)
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.umap(adata, color=["domain", "ground_truth"], title=['Scanpy (ARI=%.2f)'%ARI_score, "Ground Truth"],palette=rainbow_hex[:cluster_num],show=False)
    plt.savefig(dir+'/images/' + name + '_umap.svg', dpi=300, bbox_inches='tight')
    plt.close()

    # paga
    sc.tl.paga(adata, groups='ground_truth')
    plt.rcParams["figure.figsize"] = (4,3)
    sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20,palette=rainbow_hex[:cluster_num], 
                   title=name+'_Scanpy', legend_fontoutline=2, show=False)
    plt.savefig(dir+'/images/' + name + '_paga.svg', dpi=300, bbox_inches='tight')
    plt.close()
    label_df = adata.obs['domain']
    label_df.to_csv(dir+'/images/' + name + '_label.csv')


if __name__ == '__main__':
    names = ['151507','151508','151509','151510','151669', '151670', '151671', '151672', '151673', '151674', '151675','151676']
    # names = ['151675']
    for name in tqdm(names):
        train(name)
    df = pd.DataFrame(result)
    df.to_csv(os.path.dirname(__file__)+'/images/metric.csv')
    print('All done!')
