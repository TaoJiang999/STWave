import warnings
warnings.filterwarnings('ignore')
# import squidpy as sq
import sys
sys.path.append('/home/cavin/jt/python/SpaceFlow-master')
from SpaceFlow import SpaceFlow
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import os
result = {}
def main(name):
    path = '/home/guo/jiangtao/data/DLPFC/'
    path = '/home/cavin/jt/spatial_data/stDCL/DLPFC/'
    adata = sc.read_visium(path + name, count_file='filtered_feature_bc_matrix.h5')
    print("adata:", adata)
    import pandas as pd
    ground_truth_df = pd.read_csv(path + name + '_truth.txt', delimiter='\t', header=None, dtype=str)
    adata.obs['ground_truth'] = ground_truth_df.iloc[:, 1].values  # 假设标签在第一列
    # Data preprocessing
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var_names_make_unique()
    if sp.issparse(adata.X):
        count_matrix = adata.X.toarray()
    else:
        count_matrix = np.array(adata.X)

    sf = SpaceFlow.SpaceFlow(adata=adata)
    sf = SpaceFlow.SpaceFlow(adata=adata,count_matrix=count_matrix.all(), spatial_locs=adata.obsm['spatial'], sample_names=adata.obs_names, gene_names=adata.var_names)
    
    sf.preprocessing_data(n_top_genes=3000)

    embedding = sf.train(spatial_regularization_strength=0.1, 
             embedding_save_filepath='/home/cavin/jt/python/wave/DLPFC/_SpaceFlow/domains.tsv',
         z_dim=50, 
         lr=1e-3, 
         epochs=10, 
         max_patience=50, 
         min_stop=1000, 
         random_seed=42, 
         gpu=0, 
         regularization_acceleration=True, 
         edge_subset_sz=1000000)
    adata.obsm['emb'] = embedding
    sf.segmentation(domain_label_save_filepath="/home/cavin/jt/python/wave/DLPFC/_SpaceFlow/domains.tsv", 
                n_neighbors=50, 
                resolution=0.5)
    
    adata.obs['domain'] = np.array(sf.domains).astype(int)
    adata.obs['domain'] = adata.obs['domain'].astype(str)

    from sklearn.metrics import adjusted_rand_score
    from sklearn.metrics.cluster import normalized_mutual_info_score, homogeneity_score
    adata = adata[~adata.obs['ground_truth'].isnull()]
    uniq_pred = np.unique(adata.obs['domain'])
    print("Unique predicted domains:", uniq_pred)
    cluster_num = 5 if name in ['151669', '151670', '151671', '151672'] else 7
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

    plt.rcParams.update({'axes.titlesize':20})
    adata.obs['ground_truth'] = adata.obs['ground_truth'].astype(str)
    adata.obs['domain'] = adata.obs['domain'].astype(str)
    obs_df = adata.obs.dropna()
    NMI_score = normalized_mutual_info_score(obs_df['domain'], obs_df['ground_truth'], average_method='max')
    HS_score = homogeneity_score(obs_df['domain'], obs_df['ground_truth'])
    adata.obs['domain'] = adata.obs['domain'].copy()
    filtered_domain = adata.obs['domain'][obs_df.index]  
    filtered_ground_truth = obs_df['ground_truth']

 
    assert len(filtered_domain) == len(filtered_ground_truth), f"Shape mismatch: domain has {len(filtered_domain)} elements, ground_truth has {len(filtered_ground_truth)} elements"


    colors_num1 = adata.obs['domain'].nunique() 
    colors_num2 = adata.obs['ground_truth'].nunique()
    colors_num = max(colors_num1, colors_num2)
    ARI_score = adjusted_rand_score(filtered_domain, filtered_ground_truth)
    print("current slice:", name)
    print('ARI:', ARI_score)
    print('NMI:', NMI_score)
    print('HS:', HS_score)
    r = {'ARI':ARI_score,'NMI':NMI_score,'HS':HS_score}
    result[name] = r
    sc.pp.neighbors(adata,use_rep='emb')
    sc.pl.spatial(adata, color=["ground_truth", "domain"], palette=rainbow_hex[:colors_num], title=['ground truth', 'SpaceFlow (ARI=%.2f)' % ARI_score],show=False)
    dir = os.path.dirname(__file__)
    os.makedirs(os.path.join(dir, 'images'), exist_ok=True)
    plt.savefig(dir+'/images/' + name + '.svg', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot UMAP
    sc.tl.umap(adata)
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.umap(adata, color=["domain", "ground_truth"], title=['SpaceFlow (ARI=%.2f)'%ARI_score, "Ground Truth"],palette=rainbow_hex[:colors_num],show=False)
    plt.savefig(dir+'/images/' + name + '_umap.svg', dpi=300, bbox_inches='tight')
    plt.close()

    # paga
    sc.tl.paga(adata, groups='ground_truth')
    plt.rcParams["figure.figsize"] = (4,3)
    sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20,palette=rainbow_hex[:colors_num], 
                   title=name+'_SpaceFlow', legend_fontoutline=2, show=False)
    plt.savefig(dir+'/images/' + name + '_paga.svg', dpi=300, bbox_inches='tight')
    plt.close()
    label_df = adata.obs['domain']
    label_df.to_csv(dir+'/images/' + name + '_label.csv')



if __name__ == "__main__":
    from tqdm import tqdm
    import pandas as pd
    names = ['151507','151508','151509','151510','151669', '151670', '151671', '151672', '151673', '151674', '151675','151676']
    # names = ['151675']
    for name in tqdm(names):
        main(name)
    df = pd.DataFrame(result)
    df.to_csv(os.path.dirname(__file__)+'/images/metric.csv')
    print('All done!')


    
   
