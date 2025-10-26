import sys
sys.path.append("/home/cavin/jt/python/SEDR")
import scanpy as sc
import pandas as pd
from sklearn import metrics
import torch

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances, adjusted_rand_score, normalized_mutual_info_score, silhouette_score, davies_bouldin_score
import os
from sklearn.metrics.cluster import homogeneity_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import SEDR
print('import finished')
random_seed = 2023
SEDR.fix_seed(random_seed)
from tqdm import tqdm
# gpu
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# path
data_root = Path('E:\\spatial\\Dataset\DLPFC_')
data_root = Path('/home/cavin/jt/spatial_data/stDCL/DLPFC')
# os.environ['R_HOME'] = '/home/cavin/anaconda3/envs/tao/lib/R'
result = {}
def main(id):
    sample_name = id
    n_clusters = 5 if sample_name in ['151669', '151670', '151671', '151672'] else 7

    adata = sc.read_visium(data_root / sample_name)
    adata.var_names_make_unique()

    df_meta = pd.read_csv(data_root / sample_name / 'metadata.tsv', sep='\t')
    adata.obs['ground_truth'] = df_meta['layer_guess']

    adata.layers['count'] = adata.X.toarray()
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.filter_genes(adata, min_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e5)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", layer='count', n_top_genes=3000)
    adata = adata[:, adata.var['highly_variable'] == True]
    sc.pp.scale(adata)

    from sklearn.decomposition import PCA  # sklearn PCA is used because PCA in scanpy is not stable.
    adata_X = PCA(n_components=200, random_state=42).fit_transform(adata.X)
    adata.obsm['X_pca'] = adata_X

    graph_dict = SEDR.graph_construction(adata, 12)
    print(graph_dict)

    sedr_net = SEDR.Sedr(adata.obsm['X_pca'], graph_dict, mode='clustering', device=device)
    using_dec = True
    if using_dec:
        sedr_net.train_with_dec(N=1)
    else:
        sedr_net.train_without_dec(N=1)
    sedr_feat, _, _, _ = sedr_net.process()
    adata.obsm['SEDR'] = sedr_feat
    SEDR.mclust_R(adata, n_clusters, use_rep='SEDR', key_added='SEDR')

    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    name = id
    obs_df = adata.obs.dropna()
    NMI_score = normalized_mutual_info_score(obs_df['SEDR'], obs_df['ground_truth'], average_method='max')
    HS_score = homogeneity_score(obs_df['SEDR'], obs_df['ground_truth'])
    adata.obs['domain'] = adata.obs['SEDR'].copy()
    # new_type = refine_label(adata, radius=10, key='domain')
    # adata.obs['domain'] = new_type
    filtered_domain = adata.obs['domain'][obs_df.index]  # 按照obs_df的索引过滤domain
    filtered_ground_truth = obs_df['ground_truth']
    assert len(filtered_domain) == len(
        filtered_ground_truth), f"Shape mismatch: domain has {len(filtered_domain)} elements, ground_truth has {len(filtered_ground_truth)} elements"
    ARI_score = adjusted_rand_score(filtered_domain, filtered_ground_truth)
    print('current slice:', name)
    print('ARI:', ARI_score)
    print('NMI:', NMI_score)
    print('HS:', HS_score)
    r = {'ARI':ARI_score,'NMI':NMI_score,'HS':HS_score}
    result[name] = r
    cluster_num = 5 if name in ['151669', '151670', '151671', '151672'] else 7
    rainbow_hex = [
    '#FF6666',  # 鲜红
    '#FFB266',  # 橙黄
    '#FFFF99',  # 淡黄
    '#99FF99',  # 亮绿
    '#99FFFF',  # 青色
    '#99CCFF',  # 浅蓝
    '#C299FF'   # 紫色
    ]
    tqdm.write('saving plot')
    adata.uns['ground_truth_colors'] = rainbow_hex[:cluster_num]
    adata.uns['domain_colors'] = rainbow_hex[:cluster_num]
    plt.rcParams.update({'axes.titlesize': 20})
    sc.pl.spatial(adata, color=["ground_truth", "domain"], title=['ground truth', 'SEDR (ARI=%.2f)' % ARI_score],show=False)
    file_dir = os.path.dirname(__file__)
    # dir = trainer.wavelet+'_'+str(trainer.level)
    # dir = '/home/waas/18161127346/python/wave/DLPFC/ours/images'
    dir = os.path.dirname(__file__)
    # os.makedirs(file_dir+'/DLPFC_final/'+dir, exist_ok=True)
    # plt.savefig(file_dir+'/DLPFC_final/'+dir+'/'+name+'.png', bbox_inches='tight', dpi=300)
    plt.savefig(dir+'/images/'+name+'.svg', bbox_inches='tight', dpi=300)
    plt.close()  # 关闭当前图像，防止显示

    # Plot UMAP
    sc.pp.neighbors(adata, use_rep='SEDR')
    sc.tl.umap(adata)
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.umap(adata, color=["domain", "ground_truth"], title=['SEDR (ARI=%.2f)'%ARI_score, "Ground Truth"],palette=rainbow_hex[:cluster_num],show=False)
    plt.savefig(dir+'/images/'+ name + '_umap.svg', dpi=300, bbox_inches='tight')
    plt.close()
    # paga
    sc.tl.paga(adata, groups='ground_truth')
    plt.rcParams["figure.figsize"] = (4,3)
    sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20,palette=rainbow_hex[:cluster_num], 
                title=name+'_SEDR', legend_fontoutline=2, show=False)
    plt.savefig(dir+'/images/'+ name + '_paga.svg', dpi=300, bbox_inches='tight')
    plt.close()
    label_df = adata.obs['domain']
    label_df.to_csv(dir+'/images/' + str(id) + '_label.csv')




if __name__ == '__main__':
    data_id = ['151507', '151508', '151509', '151510', '151669', '151670',
               '151671', '151672', '151673','151674', '151675', '151676']

    ari = {}
    nmi = {}
    for id in data_id:
        main(id)
    df = pd.DataFrame(result)
    df.to_csv(os.path.dirname(__file__)+'/images/metric.csv')

    