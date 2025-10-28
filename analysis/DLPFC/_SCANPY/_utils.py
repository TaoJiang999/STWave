import dgl
import random
import warnings
import numpy as np
import sklearn
import torch
from numpy.linalg import norm
from typing import Union
import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Literal
import ot
from svg_select import rank_gene_smooth,low_pass_enhancement

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    dgl.random.seed(seed)



def clear_warnings(func, category=FutureWarning):
    def warp(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=category)
            temp = func(*args, **kwargs)
            return temp
    return warp


def select_device(GPU: Union[bool, str] = True,):
    if GPU:
        if torch.cuda.is_available():
            if isinstance(GPU, str):
                device = torch.device(GPU)
            else:
                device = torch.device('cuda:0')
        else:
            print("GPU isn't available, and use CPU to train Docs.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    return device

def prefilter_genes(adata, min_counts=None, max_counts=None, min_cells=3, max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp = np.asarray([True] * adata.shape[1], dtype=bool)
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, min_cells=min_cells)[0]) if min_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, max_cells=max_cells)[0]) if max_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, min_counts=min_counts)[0]) if min_counts is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, max_counts=max_counts)[0]) if max_counts is not None else id_tmp
    adata._inplace_subset_var(id_tmp)

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb', random_seed=2023):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    np.random.seed(random_seed)
    from sklearn.decomposition import PCA
    dim = adata.obsm[used_obsm].shape[1]
    if dim > 50:
        dim = 50
    
    pca = PCA(n_components=dim, random_state=random_seed)
    embedding = pca.fit_transform(adata.obsm[used_obsm].copy())
    adata.obsm['emb_pca'] = embedding

    import rpy2.robjects as robjects
    robjects.r.library('mclust')

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    # res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm['emb_pca']), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def svg(adata, svg_method:Literal['gft', 'gft_top', 'seurat', 'seurat_v3', 'cell_ranger', 'mix']='gft_top', n_top=3000, csvg=0.0001, smoothing=True):
    """
    Select spatially variable genes using six methods, including 'gft', 'gft_top',
    'seurat', 'seurat_v3', 'cell_ranger' and 'mix'.

    Args:
        adata: anndata
            AnnData object of scanpy package.
        svg_method: str, optional
            Methods for selecting spatially variable genes. Teh default is 'gft_top'.
        n_top: int, optional
            Number of spatially variable genes selected. The default is 3000.
        csvg: float, optional
            Smoothing coefficient of GFT for noise reduction. The default is 0.0001.
        smoothing: bool, optional
            Determine whether it is smooth for noise reduction. The default is True.

    Returns:
        adata: anndata
            AnnData object of scanpy package after choosing svgs and smoothing.
        adata_raw: anndata
            AnnData object of scanpy package before choosing svgs and smoothing.
    """
    assert svg_method in ['gft', 'gft_top', 'seurat', 'seurat_v3', 'cell_ranger', 'mix']
    if svg_method == 'seurat_v3':
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=n_top)
        adata = adata[:, adata.var['highly_variable']]
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    elif svg_method == 'mix':
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=int(n_top / 2))
        seuv3_list = adata.var_names[adata.var['highly_variable']]
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        gene_df = rank_gene_smooth(adata,
                                   spatial_info=['array_row', 'array_col'],
                                   ratio_low_freq=1,
                                   ratio_high_freq=1,
                                   ratio_neighbors=1,
                                   filter_peaks=True,
                                   S=6)
        svg_list = gene_df.index[:(n_top - len(seuv3_list))]
        merged_gene_list = np.union1d(seuv3_list, svg_list)
        adata = adata[:, merged_gene_list]
        if smoothing:
            low_pass_enhancement(adata,
                                 ratio_low_freq=15,
                                 c=csvg,
                                 spatial_info='spatial',
                                 ratio_neighbors=0.3,
                                 inplace=True)
    else:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        if svg_method == 'seurat':
            sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=n_top)
            adata = adata[:, adata.var['highly_variable']]
        elif svg_method == 'cell_ranger':
            sc.pp.highly_variable_genes(adata, flavor='cell_ranger', n_top_genes=n_top)
            adata = adata[:, adata.var['highly_variable']]
        elif svg_method == 'gft' or svg_method == 'gft_top':
            gene_df = rank_gene_smooth(adata,
                                       spatial_info='spatial',
                                       ratio_low_freq=1,
                                       ratio_high_freq=1,
                                       ratio_neighbors=1,
                                       filter_peaks=True,
                                       S=6)
            if svg_method == 'gft':
                svg_list = gene_df[gene_df.cutoff_gft_score][gene_df.qvalue < 0.05].index.tolist()
            elif svg_method == 'gft_top':
                svg_list = gene_df.index[:n_top].tolist()
            adata = adata[:, svg_list]
            adata_raw = adata.copy()
            if smoothing:
                low_pass_enhancement(adata,
                                     ratio_low_freq=15,
                                     c=csvg,
                                     spatial_info='spatial',
                                     ratio_neighbors=0.3,
                                     inplace=True)
    if(svg_method in ['gft', 'gft_top']):
        return adata, adata_raw
    return adata

def obtain_spotnet(adata, rad_cutoff=150, k_cutoff=6, knn_method='KNN', prune=True):
    """
    Constructing a graph of spots using KNN or Radius methods.

    Args:
        adata: anndata
            AnnData object of scanpy package.
        rad_cutoff: int, optional
            Truncation radius of spots. The default is 150.
        k_cutoff: int, optional
            Number of adjacent spots of a spot. The default is 6.
        knn_method: str, optional
            The method of constructing a graph of points, including 'KNN' and 'Radius'. The default is 'KNN'.
        prune: bool, optional
            Determine whether to prune or not. The default is True.

    Returns:
        None
    """
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff, n_jobs=-1) if knn_method == 'Radius' else sklearn.neighbors.NearestNeighbors(
        n_neighbors=k_cutoff + 1, n_jobs=-1)
    nbrs.fit(coor)
    if knn_method == 'Radius':
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        
    elif knn_method == 'KNN':
        distances, indices = nbrs.kneighbors(coor)
        if prune:
            non_zero_mask = distances > 0
            non_zero_distances = np.where(non_zero_mask, distances, np.nan)
            means = np.nanmean(non_zero_distances, axis=1, keepdims=True)
            stds = np.nanstd(non_zero_distances, axis=1, keepdims=True)
            boundaries = means + stds
            mask = distances > boundaries
            distances[mask] = 0
    cell_indices = np.repeat(np.arange(len(coor)), np.fromiter((len(idx) for idx in indices), dtype=int))
    
    neighbor_indices = np.concatenate(indices)
    neighbor_distances = np.concatenate(distances)
    KNN_df = pd.DataFrame({'Cell1': cell_indices, 'Cell2': neighbor_indices, 'Distance': neighbor_distances})
    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net[Spatial_Net['Distance'] > 0]

    spot_net = csr_matrix((Spatial_Net['Distance'].values > 0, Spatial_Net[['Cell1', 'Cell2']].values.T)).toarray()
    spot_net = pd.DataFrame(spot_net.astype(int), index=adata.obs_names, columns=adata.obs_names)
    adata.uns['spotnet_adj'] = spot_net
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    adata.uns['spotnet'] = Spatial_Net

def obtain_pre_spotnet(adata, adata_raw, pre_feature=False, k_cutoff=6, res_pre=0.6):
    """
    Obtain a graph of spots through pre-clustering.

    Args:
        adata: anndata
            AnnData object of scanpy package.
        adata_raw: anndata.
            Raw annData object of scanpy package.
        pre_feature: bool, optional
            Determine whether to use features to obtain a graph of spots. The default is False.
        k_cutoff: int, optional
            Number of adjacent points in constructing a graph using features. The default is 6.
        res_pre: float, optional
            Resolution value for clustering. The default is 1.0.

    Returns:
        None
    """
    # adata.obsm['emb'] = adata.uns['signal'].copy()
    # sc.pp.neighbors(adata, use_rep='emb')
    # raw = sc.AnnData(adata.uns['raw'].todence(), obs=adata.obs_names).
    raw = adata_raw.copy()
    sc.pp.pca(raw, svd_solver='arpack')
    sc.pp.neighbors(raw)
    sc.tl.louvain(raw, resolution=res_pre, key_added='expression_louvain_label')
    # new_type = refine_label(raw, radius=50, key='expression_louvain_label')
    # raw.obs['expression_louvain_label'] = new_type
    # sc.pl.spatial(raw, color=['expression_louvain_label'], title=[1], save='louvain' + '.png')
    pre_cluster = 'expression_louvain_label'
    prune_G_df = prune_spatial_Net(adata.uns['spotnet'].copy(), raw.obs[pre_cluster])
    adata.uns['spotnet_cluster'] = prune_G_df
    del adata.uns['spotnet_cluster']['Cell1_label']
    del adata.uns['spotnet_cluster']['Cell2_label']

    if pre_feature:
        feature = adata.uns['signal'].copy()
        coor = pd.DataFrame(adata.obsm['spatial'])
        coor.index = adata.obs.index
        coor.columns = ['imagerow', 'imagecol']
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            cos_sim = cosine(feature.iloc[indices[it, 1:]].values, feature.iloc[it].values)
            cos_sim = cos_sim * (cos_sim >= np.mean(cos_sim[1:]))
            KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], cos_sim)))
        KNN_df = pd.concat(KNN_list)
        KNN_df.columns = ['Cell1', 'Cell2', 'Cosine_similarity']
        Spatial_Net = KNN_df.copy()
        Spatial_Net = Spatial_Net.loc[Spatial_Net['Cosine_similarity'] > 0,]
        id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
        Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
        Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
        adata.uns['spotnet_feature'] = Spatial_Net


def prune_spatial_Net(Graph_df, label):
    pro_labels_dict = dict(zip(label.index, label))
    Graph_df['Cell1_label'] = Graph_df['Cell1'].map(pro_labels_dict)
    Graph_df['Cell2_label'] = Graph_df['Cell2'].map(pro_labels_dict)
    Graph_df = Graph_df.loc[Graph_df['Cell1_label'] == Graph_df['Cell2_label'],]
    return Graph_df

def torch2array(x):
    """Convert elements in x into arrays"""
    array = []
    for i in x:
        array.append(i.cpu().detach().numpy())
    return array

def refine_label(adata, radius=50, key='label'):
    """Used for smoothing clustering results"""
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    # adata.obs['label_refined'] = np.array(new_type)
    return new_type

def cosine(A, B):
    return np.dot(A, B) / (norm(A, axis=1) * norm(B))