import os
import sys
sys.path.append('/home/cavin/jt/python/DeepGFT-main')

current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")
from matplotlib import pyplot as plt
from DeepGFT.utils import *
from DeepGFT.genenet import obtain_genenet
from DeepGFT.train import *
import torch
import scanpy as sc
import warnings
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score, homogeneity_score
#os.environ['R_HOME'] = 'D:\\R-4.4.2'
warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed_all(2023)

result = {}
def train(name):
    
    # Read data
    #path = 'E:\\spatial_data\\stDCL\\DLPFC\\'
    path = '/home/guo/jiangtao/data/DLPFC/'
    path = '/home/cavin/jt/spatial_data/stDCL/DLPFC/'
    adata = sc.read_visium(path + name, count_file='filtered_feature_bc_matrix.h5')
    print("adata:",adata)
    #ground_truth = sc.read_csv(path + 'data/10x_Visium/DLPFC/ground_truth/' + name + '_annotation.csv', dtype='str')
    #ground_truth = sc.read_csv(path+name+'_truth.txt',delimiter='\t',dtype=str,first_column_names=None)
    import pandas as pd
    ground_truth_df = pd.read_csv(path+name+'_truth.txt', delimiter='\t', header=None, dtype=str)
    adata.obs['ground_truth'] = ground_truth_df.iloc[:, 1].values  # 
    #adata.obs['ground_truth'] = ground_truth.X

    # Data preprocessing
    adata.var_names_make_unique()
    prefilter_genes(adata, min_cells=3)
    adata, adata_raw = svg(adata, svg_method='gft_top', n_top=3000)
    # adata, adata_raw = svg(adata, svg_method='seurat_v3', n_top=3000)
    # Build spotnet and genenet
    obtain_spotnet(adata, knn_method='Radius', rad_cutoff=150)
    gene_freq_mtx, gene_eigvecs_T, gene_eigvals = f2s_gene(adata, gene_signal=1500, c1=0.05)
    obtain_genenet(adata)
    spot_freq_mtx, spot_eigvecs_T, spot_eigvals = f2s_spot(adata, spot_signal=1500, middle=3, c2=0.005)
    obtain_pre_spotnet(adata, adata_raw)

    res, lamda, emb_spot, _, attention = train_spot(adata, gene_freq_mtx, gene_eigvecs_T, spot_freq_mtx, spot_eigvecs_T,
                                                alpha=20, device=device, epoch_max=600)
    #os.environ['R_HOME'] = 'D:\\R-4.4.2'
    adata.obsm['emb'] = emb_spot
    cluster_num = 5 if name in ['151669', '151670', '151671', '151672'] else 7
    adata = mclust_R(adata, used_obsm='emb', num_cluster=cluster_num)
    obs_df = adata.obs.dropna()
    NMI_score = normalized_mutual_info_score(obs_df['mclust'], obs_df['ground_truth'], average_method='max')
    HS_score = homogeneity_score(obs_df['mclust'], obs_df['ground_truth'])
    adata.obs['domain'] = adata.obs['mclust'].copy()
    # new_type = refine_label(adata, radius=30, key='domain')
    # adata.obs['domain'] = new_type

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
    rainbow_hex = [
    '#FF6666',  # 鲜红
    '#FFB266',  # 橙黄
    '#FFFF99',  # 淡黄
    '#99FF99',  # 亮绿
    '#99FFFF',  # 青色
    '#99CCFF',  # 浅蓝
    '#C299FF'   # 紫色
    ]

    adata = adata[~adata.obs['ground_truth'].isnull()]
    adata.uns['ground_truth_colors'] = rainbow_hex[:cluster_num]
    adata.uns['domain_colors'] = rainbow_hex[:cluster_num]
    plt.rcParams.update({'axes.titlesize':20})
    adata.obs['ground_truth'] = adata.obs['ground_truth'].astype(str)
    adata.obs['domain'] = adata.obs['domain'].astype(str)
    sc.pl.spatial(adata, color=["ground_truth", "domain"], palette=rainbow_hex[:cluster_num], title=['ground truth', 'DeepGFT(ARI=%.2f)' % ARI_score],show=False)
    dir = os.path.dirname(__file__)
    # dir = os.path.dirname(dir)
    os.makedirs(os.path.join(dir, 'images'), exist_ok=True)
    plt.savefig(dir+'/images/' + name + '.svg')
    plt.close()
    # Plot UMAP
    sc.pp.neighbors(adata, use_rep='emb')
    sc.tl.umap(adata)
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.umap(adata, color=["domain", "ground_truth"], title=['DeepGFT (ARI=%.2f)'%ARI_score, "Ground Truth"],palette=rainbow_hex[:cluster_num],show=False)
    plt.savefig(dir+'/images/' + name + '_umap.svg')
    plt.close()
    # paga
    sc.tl.paga(adata, groups='ground_truth')
    plt.rcParams["figure.figsize"] = (4,3)
    sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20,palette=rainbow_hex[:cluster_num], 
                   title=name+'_DeepGFT', legend_fontoutline=2, show=False)
    plt.savefig(dir+'/images/' + name + '_paga.svg')
    plt.close()
    label_df = adata.obs['domain']
    label_df.to_csv(dir+'/images/' + name + '_label.csv')



if __name__ == '__main__':
    import pandas as pd
    names = ['151507','151508','151509','151510','151669', '151670', '151671', '151672', '151673', '151674', '151675','151676']
    # names = ['151510','151669', '151670', '151671', '151672', '151673', '151674', '151675','151676',]
    for name in names:
        train(name)
    df = pd.DataFrame(result)
    df.to_csv(os.path.dirname(__file__)+'/images/metric.csv')
    print("All slices processed successfully!")
    