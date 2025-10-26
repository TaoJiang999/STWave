import torch
import argparse
import random
import numpy as np
import pandas as pd
import sys

sys.path.append(r"/home/cavin/jt/python/conST")
from src.graph_func import graph_construction
from src.utils_func import mk_dir, adata_preprocess, load_ST_file, res_search_fixed_clus, plot_clustering
from src.training import conST_training
print('import finished')
import anndata
from sklearn import metrics
import matplotlib.pyplot as plt
import scanpy as sc
import os
import warnings
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics import pairwise_distances, adjusted_rand_score, normalized_mutual_info_score, silhouette_score, davies_bouldin_score
warnings.filterwarnings('ignore')

result = {}

parser = argparse.ArgumentParser()
parser.add_argument('--k', type=int, default=10, help='parameter k in spatial graph')
parser.add_argument('--knn_distanceType', type=str, default='euclidean',
                    help='graph distance type: euclidean/cosine/correlation')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--cell_feat_dim', type=int, default=300, help='Dim of PCA')
parser.add_argument('--feat_hidden1', type=int, default=100, help='Dim of DNN hidden 1-layer.')
parser.add_argument('--feat_hidden2', type=int, default=20, help='Dim of DNN hidden 2-layer.')
parser.add_argument('--gcn_hidden1', type=int, default=32, help='Dim of GCN hidden 1-layer.')
parser.add_argument('--gcn_hidden2', type=int, default=8, help='Dim of GCN hidden 2-layer.')
parser.add_argument('--p_drop', type=float, default=0.2, help='Dropout rate.')
parser.add_argument('--use_img', type=bool, default=False, help='Use histology images.')
parser.add_argument('--img_w', type=float, default=0.1, help='Weight of image features.')
parser.add_argument('--use_pretrained', type=bool, default=False, help='Use pretrained weights.')
parser.add_argument('--using_mask', type=bool, default=False, help='Using mask for multi-dataset.')
parser.add_argument('--feat_w', type=float, default=10, help='Weight of DNN loss.')
parser.add_argument('--gcn_w', type=float, default=0.1, help='Weight of GCN loss.')
parser.add_argument('--dec_kl_w', type=float, default=10, help='Weight of DEC loss.')
parser.add_argument('--gcn_lr', type=float, default=0.01, help='Initial GNN learning rate.')
parser.add_argument('--gcn_decay', type=float, default=0.01, help='Initial decay rate.')
parser.add_argument('--dec_cluster_n', type=int, default=10, help='DEC cluster number.')
parser.add_argument('--dec_interval', type=int, default=20, help='DEC interval nnumber.')
parser.add_argument('--dec_tol', type=float, default=0.00, help='DEC tol.')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--beta', type=float, default=100, help='beta value for l2c')
parser.add_argument('--cont_l2l', type=float, default=0.3, help='Weight of local contrastive learning loss.')
parser.add_argument('--cont_l2c', type=float, default=0.1, help='Weight of context contrastive learning loss.')
parser.add_argument('--cont_l2g', type=float, default=0.1, help='Weight of global contrastive learning loss.')

parser.add_argument('--edge_drop_p1', type=float, default=0.1, help='drop rate of adjacent matrix of the first view')
parser.add_argument('--edge_drop_p2', type=float, default=0.1, help='drop rate of adjacent matrix of the second view')
parser.add_argument('--node_drop_p1', type=float, default=0.2, help='drop rate of node features of the first view')
parser.add_argument('--node_drop_p2', type=float, default=0.3, help='drop rate of node features of the second view')

# ______________ Eval clustering Setting ______________
parser.add_argument('--eval_resolution', type=int, default=1, help='Eval cluster number.')
parser.add_argument('--eval_graph_n', type=int, default=20, help='Eval graph kN tol.')

params = parser.parse_args(args=['--k', '20', '--knn_distanceType', 'euclidean', '--epochs', '200'])

np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Using device: ' + device)
params.device = device

# set seed before every run
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# read related files
# read dataset
def main(id, domain):
    file_path = r"/home/cavin/jt/spatial_data/stDCL/DLPFC/"
    section_id = id
    adata = sc.read_visium(path=os.path.join(file_path, section_id),
                           count_file="filtered_feature_bc_matrix.h5",
                           library_id=section_id,
                           source_image_path=os.path.join(file_path, section_id, "spatial"))
    n_clusters = domain
    Ann_df = pd.read_csv(os.path.join(file_path, section_id + '_truth.txt'), sep='\t', header=None,
                         index_col=0)
    Ann_df.columns = ['ground_truth']
    adata.obs['ground_truth'] = Ann_df.loc[adata.obs_names, 'ground_truth']
    adata = adata[~pd.isnull(adata.obs['ground_truth'])]

    adata_X = sc.pp.normalize_total(adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
    adata_X = sc.pp.scale(adata_X)
    adata_X = sc.pp.pca(adata_X, n_comps=300)
    graph_dict = graph_construction(adata.obsm['spatial'], adata.shape[0], params)
    params.cell_num = adata.shape[0]
    ari_r = 0
    nmi_r = 0
    for i in range(10, 11):
        params.seed = i
        seed_torch(params.seed)
        if params.use_img:
            img_transformed = np.load('./MAE-pytorch/extracted_feature.npy')
            img_transformed = (img_transformed - img_transformed.mean()) / img_transformed.std() * adata_X.std() + adata_X.mean()
            conST_net = conST_training(adata_X, graph_dict, params, n_clusters, img_transformed)
        else:
            conST_net = conST_training(adata_X, graph_dict, params, n_clusters)
        if params.use_pretrained:
            conST_net.load_model('conST_151673.pth')
        else:
            conST_net.pretraining()
            conST_net.major_training()

        conST_embedding = conST_net.get_embedding()

        # np.save(f'{params.save_path}/conST_result.npy', conST_embedding)
        # clustering
        #adata_conST = anndata.AnnData(conST_embedding)
        adata.obsm['emb'] = conST_embedding
        #adata_conST.obs_names = adata.obs_names
        # adata_conST.uns['spatial'] = adata_h5.uns['spatial']
        #adata_conST.obsm['spatial'] = adata_h5.obsm['spatial']
        #adata_conST.obs['Ground Truth'] = adata.obs['Ground Truth']
        

        sc.pp.neighbors(adata, use_rep='emb')

        eval_resolution = res_search_fixed_clus(adata, n_clusters)
        print(eval_resolution)
        sc.tl.leiden(adata, resolution=eval_resolution)
        print(adata)

        plot_color = ["#d62728", "#9467bd", "#e377c2", "#8c564b", "#ff7f0e", "#2ca02c", "#1f77b4"]
        rainbow_hex = [
        '#FF6666',  # 鲜红
        '#FFB266',  # 橙黄
        '#FFFF99',  # 淡黄
        '#99FF99',  # 亮绿
        '#99FFFF',  # 青色
        '#99CCFF',  # 浅蓝
        '#C299FF'   # 紫色
        ]
        obs_df = adata.obs.dropna()
        ARI = adjusted_rand_score(obs_df['leiden'], obs_df['ground_truth'])
        ARI_score = ARI
        print('Adjusted rand index = %.4f' % ARI)
        NMI = normalized_mutual_info_score(obs_df['ground_truth'], obs_df['leiden'])  # 计算nmi
        print('NMI = %.4f' % NMI)
        HS_score = homogeneity_score(obs_df['leiden'], obs_df['ground_truth'])
        print('HS:', HS_score)
        r = {'ARI':ARI_score,'NMI':NMI,'HS':HS_score}
        
        name = id
        result[name] = r
        cluster_num = 5 if name in ['151669', '151670', '151671', '151672'] else 7
        dir = os.path.dirname(__file__)
        adata.obs['domain'] = adata.obs['leiden'].astype('category')
        plt.rcParams.update({'axes.titlesize': 20})
        sc.pl.spatial(adata, color=["ground_truth", "domain"], palette=rainbow_hex[:cluster_num], title=['ground truth', 'ConST (ARI=%.2f)' % ARI_score],show=False)
        
        os.makedirs(dir+'/images', exist_ok=True)
        
        plt.savefig(dir+'/images/'+name+'.svg')
        plt.close()  # 关闭当前图像，防止显示

        # Plot UMAP
        # sc.pp.neighbors(adata, use_rep='emb')
        sc.tl.umap(adata)
        plt.rcParams["figure.figsize"] = (3, 3)
        sc.pl.umap(adata, color=["domain", "ground_truth"], title=['ConST (ARI=%.2f)'%ARI_score, "Ground Truth"],palette=rainbow_hex[:cluster_num],show=False)
        plt.savefig(dir+'/images/'+ name + '_umap.svg')
        plt.close()
        # paga
        sc.tl.paga(adata, groups='ground_truth')
        plt.rcParams["figure.figsize"] = (4,3)
        sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20,palette=rainbow_hex[:cluster_num], 
                        title=name+'_ConST', legend_fontoutline=2, show=False)
        plt.savefig(dir+'/images/' + name + '_paga.svg')
        plt.close()
        label_df = adata.obs['domain']
        label_df.to_csv(dir+'/images/' + name + '_label.csv')

        
        ari_r = ARI
        nmi_r = NMI
        
        print('saved')
    return ari_r, nmi_r

if __name__ == '__main__':
    data_id = ['151507', '151508', '151509', '151510', '151669', '151670',
               '151671', '151672', '151673', '151674', '151675', '151676']
    domains = [7, 7, 7, 7, 5, 5, 5, 5, 7, 7, 7, 7]

    #data_id = ['151507','151508']
    #domains = [7, 7]
    ari = {}
    nmi = {}
    for id, domain in zip(data_id[:], domains[:]):
        ari[id], nmi[id] = main(id, domain)
    df = pd.DataFrame(result)
    df.to_csv(os.path.dirname(__file__)+'/images/metric.csv')