import os
import sys
sys.path.append('/home/cavin/jt/python/MuCoST-master')
import pandas as pd
import warnings
from tqdm import tqdm
# 忽略 ImplicitModificationWarning

warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
import scanpy as sc
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
from MuCoST.mucost import training_model
from MuCoST.utils import mclust
from MuCoST.config import set_arg
from sklearn.metrics.cluster import  homogeneity_score
from sklearn.metrics import pairwise_distances, adjusted_rand_score, normalized_mutual_info_score, silhouette_score, davies_bouldin_score
result = {}
def main(id, domain):
    opt = set_arg()
    arg = opt.parse_args(['--mode_his', 'noh'])
    arg.n_domain=domain
    arg.epoch = 1000

    section_id = id
    input_dir = os.path.join('/home/cavin/jt/spatial_data/stDCL/DLPFC', section_id)
    adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()
    # histology mode
    # img=cv2.imread(input_dir + "/spatial/full_image.tif")
    # adata.uns['image']=img

    Ann_df = pd.read_csv(os.path.join('/home/cavin/jt/spatial_data/stDCL/DLPFC', section_id+'_truth.txt'), sep='\t', header=None, index_col=0)
    Ann_df.columns = ['ground_truth']
    adata.obs['ground_truth'] = Ann_df.loc[adata.obs_names, 'ground_truth']

    training_model(adata, arg)

    adata = mclust(adata, arg, refine=False)

    adata = adata[~pd.isnull(adata.obs['ground_truth'])]
    obs_df = adata.obs.dropna()
    ARI = adjusted_rand_score(obs_df['mclust'], obs_df['ground_truth'])
    print('Adjusted rand index = %.4f' %ARI)
    NMI = normalized_mutual_info_score(obs_df['ground_truth'], obs_df['mclust'])#计算nmi
    print('NMI = %.4f' %NMI)
    HS_score = homogeneity_score(obs_df['mclust'], obs_df['ground_truth'])
    r = {'ARI':ARI,'NMI':NMI,'HS':HS_score}
    result[id] = r
    rainbow_hex = [
        '#FF6666',  # 鲜红
        '#FFB266',  # 橙黄
        '#FFFF99',  # 淡黄
        '#99FF99',  # 亮绿
        '#99FFFF',  # 青色
        '#99CCFF',  # 浅蓝
        '#C299FF'  # 紫色
    ]
    cluster_num = domain
    adata.obs['domain'] = adata.obs['mclust'].copy()
    adata = adata[~adata.obs['ground_truth'].isnull()]
    tqdm.write('saving plot')
    adata.uns['ground_truth_colors'] = rainbow_hex[:cluster_num]
    adata.uns['domain_colors'] = rainbow_hex[:cluster_num]
    plt.rcParams.update({'axes.titlesize': 20})
    sc.pl.spatial(adata, color=["ground_truth", "domain"], title=['ground truth', 'MuCost(ARI=%.2f)' % ARI],
                  show=False)
    file_dir = os.path.dirname(__file__)
    # dir = trainer.wavelet+'_'+str(trainer.level)
    # dir = '/home/waas/18161127346/python/wave/DLPFC/ours/images'
    # os.makedirs(file_dir+'/DLPFC_final/'+dir, exist_ok=True)
    # plt.savefig(file_dir+'/DLPFC_final/'+dir+'/'+name+'.png', bbox_inches='tight', dpi=300)
    plt.savefig(file_dir + '/images/' + id + '.svg', bbox_inches='tight', dpi=300)
    plt.close()  # 关闭当前图像，防止显示

    # Plot UMAP
    sc.pp.neighbors(adata, use_rep='MuCoST')
    sc.tl.umap(adata)
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.umap(adata, color=["domain", "ground_truth"], title=['MuCost (ARI=%.2f)' % ARI, "ground_truth"],
               palette=rainbow_hex[:cluster_num], show=False)
    plt.savefig(file_dir + '/images/' + id + '_umap.svg', dpi=300, bbox_inches='tight')
    plt.close()
    # paga
    sc.tl.paga(adata, groups='ground_truth')
    plt.rcParams["figure.figsize"] = (4, 3)
    sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20, palette=rainbow_hex[:cluster_num],
                       title=id + '_MuCost', legend_fontoutline=2, show=False)
    plt.savefig(file_dir + '/images/' + id + '_paga.svg', dpi=300, bbox_inches='tight')
    plt.close()
    label_df = adata.obs['domain']
    label_df.to_csv(file_dir+'/images/' + id + '_label.csv')


if __name__ == '__main__':
    data_id = ['151507', '151508', '151509', '151510', '151669', '151670',
               '151671', '151672', '151673','151674', '151675', '151676']
    domains = [7, 7, 7, 7, 5, 5, 5, 5, 7, 7, 7, 7]

    for id, domain in zip(data_id[:], domains[:]):
       main(id, domain)
    df = pd.DataFrame(result)
    df.to_csv(os.path.dirname(__file__)+'/images/metric.csv')
