# use py3.9 environment
import os
import torch
import sys
sys.path.append('/home/cavin/jt/python/GraphST-main')
import pandas as pd
import scanpy as sc
from sklearn import metrics
import matplotlib.pyplot as plt
import multiprocessing as mp
from GraphST import GraphST
from sklearn.metrics.cluster import  homogeneity_score
from sklearn.metrics import pairwise_distances, adjusted_rand_score, normalized_mutual_info_score, silhouette_score, davies_bouldin_score
import warnings
from tqdm import tqdm
from IPython.display import display
result = {}

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# the location of R, which is necessary for mclust algorithm. Please replace the path below with local R installation
# path
# os.environ['R_HOME'] = r'D:\R-4.4.2'
# the location of R (used for the mclust clustering)
# os.environ['R_USER'] = r'D:\anaconda\envs\mucost\Lib\site-packages\rpy2'
# read dataset
file_path = r"/home/cavin/jt/spatial_data/stDCL/DLPFC/"
def main(id, domain):
    section_id = id
    adata = sc.read_visium(path=os.path.join(file_path, section_id),
                           count_file="filtered_feature_bc_matrix.h5",
                           library_id=section_id)
    #truth = pd.read_table(os.path.join(file_path, section_id, "metadata.tsv"), index_col=0)
    # Need to remove NAN
    #truth.drop(truth[truth["layer_guess_reordered"].isna()].index, inplace=True)
    Ann_df = pd.read_csv(os.path.join(file_path, section_id + '_truth.txt'), sep='\t', header=None,
                             index_col=0)
    Ann_df.columns = ['ground_truth']
    adata.obs['ground_truth'] = Ann_df.loc[adata.obs_names, 'ground_truth']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    n_clusters = domain
    #adata = adata[truth.index, :]
    adata.var_names_make_unique()
    print(n_clusters)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    # define model
    save_obj = pd.DataFrame()
    nmi_g = 0
    ari_g = 0
    for i in range(1, 2):
        print("Now the cycle is:", i)
        model = GraphST.GraphST(adata, device=device, random_seed=i)

        # train model
        adata = model.train()
        # Only for DLPFC
        # set radius to specify the number of neighbors considered during refinement
        radius = 50
        tool = 'mclust'  # mclust, leiden, and louvain
        # Note: rpy2 is required for mclust, and rpy2 in not supported by Windows.
        # clustering
        from GraphST.utils import clustering

        if tool == 'mclust':
            clustering(adata, n_clusters, radius=radius, method=tool,
                       refinement=True)  # For DLPFC dataset, we use optional refinement step.
        elif tool in ['leiden', 'louvain']:
            clustering(adata, n_clusters, radius=radius, method=tool, start=0.3, end=3, increment=0.02, refinement=False)
        #adata.obs["leiden"]
        save_obj.index = adata.obs.index
        save_obj.index.name = "ID"
        save_obj = pd.concat([save_obj, adata.obs["mclust"]], axis=1)
        obs_df = adata.obs.dropna()
        ARI = adjusted_rand_score(obs_df['mclust'], obs_df['ground_truth'])
        print('Adjusted rand index = %.4f' % ARI)
        NMI = normalized_mutual_info_score(obs_df['ground_truth'], obs_df['mclust'])  # 计算nmi
        print('NMI = %.4f' % NMI)
        HS_score = homogeneity_score(obs_df['mclust'], obs_df['ground_truth'])
        r = {'ARI':ARI,'NMI':NMI,'HS':HS_score}
        result[id] = r
        rainbow_hex = [
            '#FF6666',  
            '#FFB266',  
            '#FFFF99',  
            '#99FF99', 
            '#99FFFF',  
            '#99CCFF',  
            '#C299FF' 
        ]
        cluster_num = domain
        adata.obs['domain'] = adata.obs['mclust'].copy()
        adata = adata[~adata.obs['ground_truth'].isnull()]
        tqdm.write('saving plot')
        adata.uns['ground_truth_colors'] = rainbow_hex[:cluster_num]
        adata.uns['domain_colors'] = rainbow_hex[:cluster_num]
        plt.rcParams.update({'axes.titlesize': 20})
        sc.pl.spatial(adata, color=["ground_truth", "domain"], title=['ground truth', 'GraphST(ARI=%.2f)' % ARI],
                      show=False)
        file_dir = os.path.dirname(__file__)
        # dir = trainer.wavelet+'_'+str(trainer.level)
        # dir = '/home/waas/18161127346/python/wave/DLPFC/ours/images'
        # os.makedirs(file_dir+'/DLPFC_final/'+dir, exist_ok=True)
        # plt.savefig(file_dir+'/DLPFC_final/'+dir+'/'+name+'.png', bbox_inches='tight', dpi=300)
        plt.savefig(file_dir + '/images/' + id + '.svg', bbox_inches='tight', dpi=300)
        plt.close()  
        # Plot UMAP
        sc.pp.neighbors(adata, use_rep='emb_pca')
        sc.tl.umap(adata)
        plt.rcParams["figure.figsize"] = (3, 3)
        sc.pl.umap(adata, color=["domain", "ground_truth"], title=['GraphST (ARI=%.2f)' % ARI, "ground_truth"],
                   palette=rainbow_hex[:cluster_num], show=False)
        plt.savefig(file_dir + '/images/' + id + '_umap.svg', dpi=300, bbox_inches='tight')
        plt.close()
        # paga
        sc.tl.paga(adata, groups='ground_truth')
        plt.rcParams["figure.figsize"] = (4, 3)
        sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20, palette=rainbow_hex[:cluster_num],
                           title=id + '_GraphST', legend_fontoutline=2, show=False)
        plt.savefig(file_dir + '/images/' + id + '_paga.svg', dpi=300, bbox_inches='tight')
        plt.close()
        label_df = adata.obs['domain']
        label_df.to_csv(file_dir+'/images/' + id + '_label.csv')



if __name__ == '__main__':
    data_id = ['151507', '151508', '151509', '151510', '151669', '151670',
               '151671', '151672', '151673', '151674', '151675', '151676']
    domains = [7, 7, 7, 7, 5, 5, 5, 5, 7, 7, 7, 7]
    ari = {}
    nmi = {}
    for id, domain in zip(data_id[:], domains[:]):
        main(id, domain)
    df = pd.DataFrame(result)
    df.to_csv(os.path.dirname(__file__)+'/images/metric.csv')
    # dfari = pd.DataFrame.from_dict(ari, orient='index')
    # dfari.to_csv(os.path.join('E:/pythonstudy/benchmark_method/graphst/DLPFC/ari', 'ari.csv'), index=True, header=True)
    # dfnmi = pd.DataFrame.from_dict(nmi, orient='index')
    # dfnmi.to_csv(os.path.join('E:/pythonstudy/benchmark_method/graphst/DLPFC/ari', 'nmi.csv'), index=True, header=True)
    # print('ari and nmi is saved')