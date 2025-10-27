import sys
sys.path.append('/home/cavin/jt/python/wave/time_no_dic/_past')
import past
import os
import scanpy as sc
import warnings
import torch
import numpy as np
import anndata as ad
import pandas as pd
# from myutils import measure_resources
import time
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score,homogeneity_score
from tqdm import tqdm
from matplotlib import pyplot as plt
result = {}
# @measure_resources
def loading_and_preprocess_data(name):
    # adata = sc.read_h5ad('/home/guo/jt/data/simulate/simu1/rep1/data.h5ad')
    # adata = sc.read_h5ad('/home/guo/jt/data/simulate/ad/data_640k.h5ad')
    # adata = adata[:5000]
    path = '/home/cavin/jt/spatial_data/stDCL/DLPFC/'
    adata = sc.read_visium(path+name)
    tqdm.write('------preprocesing data...')
    start = time.time()
    import pandas as pd
    ground_truth_df = pd.read_csv(path + name + '_truth.txt', delimiter='\t', header=None, dtype=str)
    adata.obs['ground_truth'] = ground_truth_df.iloc[:, 1].values  # 假设标签在第一列
    # Data preprocessing
    adata.var_names_make_unique()
    adata.X = adata.X.astype(np.float32)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000,subset=True)
    sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True)
    sc.pp.scale(adata,zero_center=True, max_value=10)
    adata = adata[:, adata.var.highly_variable]
    tqdm.write(f'take times:{time.time()-start}')
    # tqdm.write('------running pca...')
    # start = time.time()
    # sc.pp.pca(adata, svd_solver='arpack', n_comps=200)
    # tqdm.write(f'take times:{time.time()-start}')
    print("adata:", adata)
    return adata

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# @measure_resources
def train(adata):
    dim_reduction = 'HVG'
    if dim_reduction == 'PCA':
            if 'X_pca' not in adata.obsm.keys():
                raise ValueError("PCA has not been done! Run sc.pp.pca first!")
    elif dim_reduction == 'HVG':
            if 'highly_variable' not in adata.var.keys():
                raise ValueError("HVG has not been computed! Run sc.pp.highly_variable_genes first!")
    past.setup_seed(666)
    # sdata = past.preprocess(sdata, min_cells=3)
    # X_pca = adata.obsm['X_pca'].copy()  # shape = [5000, 50]
    # obs = adata.obs.copy()
    # var = pd.DataFrame(index=[f'PC{i+1}' for i in range(X_pca.shape[1])])  # shape = [50, 0]

    # adata_pca = ad.AnnData(X=X_pca, obs=obs, var=var)
    # adata_pca.obsm = adata.obsm.copy()
    # adata_raw = adata.copy()
    # adata = adata_pca
    # del adata_pca
    PAST_model = past.PAST(d_in=adata.X.shape[1], d_lat=30, k_neighbors=8, dropout=0.1).to(device)
    PAST_model.model_train(adata, epochs=50, lr=1e-3, device=device)
    adata = PAST_model.output(adata)
    return adata

# @measure_resources
def clustering(adata):
    adata = past.mclust_R(adata, num_cluster=adata.obs['ground_truth'].nunique(), used_obsm='embedding')
    # adata = past.default_louvain(adata, use_rep="embedding")

# @measure_resources
def save_data(adata,name):
    obs_df = adata.obs.dropna()
    NMI_score = normalized_mutual_info_score(obs_df['mclust'], obs_df['ground_truth'], average_method='max')
    HS_score = homogeneity_score(obs_df['mclust'], obs_df['ground_truth'])
    adata.obs['domain'] = adata.obs['mclust'].copy()
    filtered_domain = adata.obs['domain'][obs_df.index]  # 按照obs_df的索引过滤domain
    filtered_ground_truth = obs_df['ground_truth']
    assert len(filtered_domain) == len(
        filtered_ground_truth), f"Shape mismatch: domain has {len(filtered_domain)} elements, ground_truth has {len(filtered_ground_truth)} elements"
    ARI_score = adjusted_rand_score(filtered_domain, filtered_ground_truth)
    print('ARI:', ARI_score)
    print('NMI:', NMI_score)
    print('HS:', HS_score)

    rainbow_hex = [
    '#FF6666',  
    '#FFFF99',  
    '#99FF99',  
    '#99FFFF',  
    '#99CCFF',  
    '#C299FF'   
    ]
    adata = adata[~adata.obs['ground_truth'].isnull()]
    r = {'ARI':ARI_score,'NMI':NMI_score,'HS':HS_score}
    result[name] = r
    cluster_num = adata.obs['ground_truth'].nunique()
    tqdm.write('saving plot')
    adata.uns['ground_truth_colors'] = rainbow_hex[:cluster_num]
    adata.uns['domain_colors'] = rainbow_hex[:cluster_num]
    plt.rcParams.update({'axes.titlesize': 20})
    sc.pl.spatial(adata, color=["ground_truth", "domain"], title=['ground truth', 'PAST(ARI=%.2f)' % ARI_score],show=False)
        
    dir = os.path.dirname(__file__)+'/images/'
    label_df = adata.obs['domain']
    label_df.to_csv(dir + name + '_label.csv')
    plt.savefig(dir+name+'.svg', bbox_inches='tight', dpi=300)
    plt.close()  

    # Plot UMAP
    sc.pp.neighbors(adata, use_rep='embedding')
    sc.tl.umap(adata)
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.umap(adata, color=["domain", "ground_truth"], title=['PAST (ARI=%.2f)'%ARI_score, "Ground Truth"],palette=rainbow_hex[:cluster_num],show=False)
    plt.savefig(dir+ name + '_umap.svg', dpi=300, bbox_inches='tight')
    plt.close()
    # paga
    sc.tl.paga(adata, groups='ground_truth')
    plt.rcParams["figure.figsize"] = (4,3)
    sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20,palette=rainbow_hex[:cluster_num], 
                    title=name+'_Ours', legend_fontoutline=2, show=False)
    plt.savefig(dir + name + '_paga.svg', dpi=300, bbox_inches='tight')
    plt.close()


    

if __name__ == "__main__":
    #  import pandas as pd
     names = ['151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676']
     for name in names:
        adata = loading_and_preprocess_data(name)
        adata = train(adata)
        clustering(adata)
        save_data(adata,name)
     df = pd.DataFrame(result)
     df.to_csv(os.path.dirname(__file__)+'/images/metric.csv')