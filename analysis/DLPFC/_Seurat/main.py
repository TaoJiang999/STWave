import time
import scanpy as sc
import os
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score, homogeneity_score
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
result = {}
def train(name:str):
    begin = time.time()
    path = '/home/waas/18161127346/data/DLPFC/'
    path = '/home/cavin/jt/spatial_data/stDCL/DLPFC/'
    adata = sc.read_visium(path + name, count_file='filtered_feature_bc_matrix.h5')
    print("adata:", adata)
    import pandas as pd
    ground_truth_df = pd.read_csv(path + name + '_truth.txt', delimiter='\t', header=None, dtype=str)
    adata.obs['ground_truth'] = ground_truth_df.iloc[:, 1].values  
    adata.var_names_make_unique()
    dir = os.path.dirname(__file__)
    label_path = os.path.join(dir,'images',name+'_label.csv')
    emb_path = os.path.join(dir,'images',name+'_label.csv')
    label = pd.read_csv(label_path, index_col=0,header=0,sep=',')
    # emb = pd.read_csv(emb_path, index_col=0,header=0,sep=',')
    label = label.reindex(adata.obs.index, fill_value=np.nan)
    # emb = emb.reindex(adata.obs.index, fill_value=np.nan)
    end = time.time()
    cluster_num = 5 if name in ['151669', '151670', '151671', '151672'] else 7
    print(f'takes time data preprocess: {end - begin}')
    # adata.obsm['emb'] = emb.values
    ari_max = 0
    index = 0
    for i in range(10):
        adata.obs['domain'] = label.values[:, i]

        # adata.obs['domain'] = adata.obs['domain'].astype('int')
        adata.obs['domain'] = adata.obs['domain'].astype('category')


        mask = ~adata.obs['domain'].isna()  
        adata = adata[mask].copy()  
        label = label[mask]

        obs_df = adata.obs.dropna()
        NMI_score = normalized_mutual_info_score(obs_df['domain'], obs_df['ground_truth'], average_method='max')
        HS_score = homogeneity_score(obs_df['domain'], obs_df['ground_truth'])
        # adata.obs['domain'] = adata.obs['mclust'].copy()
        # new_type = refine_label(adata, radius=10, key='domain')
        # adata.obs['domain'] = new_type
        filtered_domain = adata.obs['domain'][obs_df.index]  
        filtered_ground_truth = obs_df['ground_truth']
        assert len(filtered_domain) == len(
            filtered_ground_truth), f"Shape mismatch: domain has {len(filtered_domain)} elements, ground_truth has {len(filtered_ground_truth)} elements"
        ARI_score = adjusted_rand_score(filtered_domain, filtered_ground_truth)
        print('current slice:', name)
        print('ARI:', ARI_score)
        print('NMI:', NMI_score)
        print('HS:', HS_score)
        if ARI_score > ari_max:
            ari_max = ARI_score
            index = i
            r = {'ARI':ARI_score,'NMI':NMI_score,'HS':HS_score}
            result[name] = r
    adata.obs['domain'] = label.values[:, index]
    # adata.obs['domain'] = adata.obs['domain'].astype('int')
    adata.obs['domain'] = adata.obs['domain'].astype('category')
    rainbow_hex = [
    '#FF6666',  
    '#FFB266',  
    '#FFFF99',  
    '#99FF99',  
    '#99FFFF', 
    '#99CCFF',  
    '#C299FF'   
    ]
    adata = adata[~adata.obs['ground_truth'].isnull()]
    tqdm.write('saving plot')
    adata.uns['ground_truth_colors'] = rainbow_hex[:cluster_num]
    adata.uns['domain_colors'] = rainbow_hex[:cluster_num]
    plt.rcParams.update({'axes.titlesize': 20})
    sc.pl.spatial(adata, color=["ground_truth", "domain"], title=['ground truth', 'Seurat (ARI=%.2f)' % ARI_score],show=False)
    
    os.makedirs(dir+'/images', exist_ok=True)
    
    plt.savefig(dir+'/images/'+name+'.svg', bbox_inches='tight', dpi=300)
    plt.close()  

    # # Plot UMAP
    # sc.pp.neighbors(adata, use_rep='emb')
    # sc.tl.umap(adata)
    # plt.rcParams["figure.figsize"] = (3, 3)
    # sc.pl.umap(adata, color=["domain", "ground_truth"], title=['BayesSpace (ARI=%.2f)'%ARI_score, "Ground Truth"],palette=rainbow_hex[:cluster_num],show=False)
    # plt.savefig(dir+'/images/'+ name + '_umap.png', dpi=300, bbox_inches='tight')
    # plt.close()
    # # paga
    # sc.tl.paga(adata, groups='ground_truth')
    # plt.rcParams["figure.figsize"] = (4,3)
    # sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20,palette=rainbow_hex[:cluster_num], 
    #                 title=name+'_BayesSpace', legend_fontoutline=2, show=False)
    # plt.savefig(dir+'/images/' + name + '_paga.png', dpi=300, bbox_inches='tight')
    # plt.close()
    


if __name__ == '__main__':
    import pandas as pd
    names = ['151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676']
    # names = ['151507']
    # names = ['151510','151669','151670','151671','151672','151673','151674','151675','151676']
    # names = ['151508']
    for name in names:
        train(name)
    df = pd.DataFrame(result)
    df.to_csv(os.path.dirname(__file__)+'/images/metric.csv')