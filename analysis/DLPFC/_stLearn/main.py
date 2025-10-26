import scanpy as sc
import stlearn as st
import pathlib as pathlib
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score, homogeneity_score
st.settings.set_figure_params(dpi=120)
import matplotlib.pyplot as plt
# Ignore all warnings
import warnings
import os
warnings.filterwarnings("ignore")

result = {}
def train(name):
    # Read data
    path = '/home/guo/jt/data/DLPFC/'
    adata = sc.read_visium(path + name, count_file='filtered_feature_bc_matrix.h5')
    # adata.uns['spatial'][name]['use_quality'] ='lowres'
    # print("adata:", adata)
    # print(adata.uns['spatial'][name].keys())
    # print(adata.uns['spatial'][name]['images'].keys())
    import pandas as pd
    ground_truth_df = pd.read_csv(path + name + '_truth.txt', delimiter='\t', header=None, dtype=str)
    adata.obs['ground_truth'] = ground_truth_df.iloc[:, 1].values  # 假设标签在第一列


    adata = st.convert_scanpy(adata)


    st.pp.filter_genes(adata, min_cells=1)
    st.pp.normalize_total(adata)
    st.pp.log1p(adata)

    # pre-processing for spot image
    st.pp.tiling(adata, out_path="./tiling")

    # this step uses deep learning model to extract high-level features from tile images
    # may need few minutes to be completed
    st.pp.extract_feature(adata)

    # run PCA for gene expression data
    st.em.run_pca(adata, n_comps=50)

    adata_sme = adata.copy()
    # apply stSME to normalise log transformed data
    st.spatial.SME.SME_normalize(adata_sme, use_data="raw")
    adata_sme.X = adata_sme.obsm['raw_SME_normalized']
    st.pp.scale(adata_sme)
    st.em.run_pca(adata_sme, n_comps=50)

    # K-means clustering on stSME normalised PCA
    cluster_num = 5 if name in ['151669', '151670', '151671', '151672'] else 7
    adata_sme = st.tl.clustering.kmeans(adata_sme, n_clusters=cluster_num, use_data="X_pca", key_added="X_pca_kmeans",copy=True)
    st.pl.cluster_plot(adata_sme, use_label="X_pca_kmeans")

    adata = adata_sme.copy()
    print('adata:',adata)
    obs_df = adata.obs.dropna()
    NMI_score = normalized_mutual_info_score(obs_df['X_pca_kmeans'], obs_df['ground_truth'], average_method='max')
    HS_score = homogeneity_score(obs_df['X_pca_kmeans'], obs_df['ground_truth'])
    adata.obs['domain'] = adata.obs['X_pca_kmeans'].copy()
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
    rainbow_hex = [
        '#FF6666',  # 鲜红
        '#FFB266',  # 橙黄
        '#FFFF99',  # 淡黄
        '#99FF99',  # 亮绿
        '#99FFFF',  # 青色
        '#99CCFF',  # 浅蓝
        '#C299FF'  # 紫色
    ]

    adata = adata[~adata.obs['ground_truth'].isnull()]
    adata.uns['ground_truth_colors'] = rainbow_hex[:cluster_num]
    adata.uns['domain_colors'] = rainbow_hex[:cluster_num]
    plt.rcParams.update({'axes.titlesize': 20})

    sc.pl.spatial(adata, color=["ground_truth", "domain"], title=['ground truth', 'stLearn (ARI=%.2f)' % ARI_score],
                  show=False)
    file_dir = os.path.dirname(__file__)

    dir = '/home/guo/jt/python/wave/DLPFC/stLearn/images'
    os.makedirs(dir, exist_ok=True)
    plt.savefig(dir + '/' + name + '.svg', bbox_inches='tight', dpi=300)
    plt.close()  # 关闭当前图像，防止显示

    # Plot UMAP
    sc.pp.neighbors(adata, use_rep='X_pca')
    sc.tl.umap(adata)
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.umap(adata, color=["domain", "ground_truth"], title=['stLearn (ARI=%.2f)' % ARI_score, "Ground Truth"],
               palette=rainbow_hex[:cluster_num], show=False)
    plt.savefig(dir + '/' + name + '_umap.svg', dpi=300, bbox_inches='tight')
    plt.close()
    # paga
    sc.tl.paga(adata, groups='ground_truth')
    plt.rcParams["figure.figsize"] = (4, 3)
    sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20, palette=rainbow_hex[:cluster_num],
                       title=name + '_stLearn', legend_fontoutline=2, show=False)
    plt.savefig(dir + '/' + name + '_paga.svg', dpi=300, bbox_inches='tight')
    plt.close()
    label_df = adata.obs['domain']
    label_df.to_csv(dir+'/' + name + '_label.csv')




if __name__ == '__main__':
    from tqdm import tqdm
    import pandas as pd
    names = ['151507','151508','151509','151510','151669', '151670', '151671', '151672', '151673', '151674', '151675','151676']
    # names = ['151675']
    for name in tqdm(names):
        train(name)
    df = pd.DataFrame(result)
    df.to_csv(os.path.dirname(__file__)+'/images/metric.csv')
    print('All done!')
