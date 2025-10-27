import os,csv,re
import pandas as pd
import numpy as np
import scanpy as sc
import math
import sys
sys.path.append('/home/cavin/jt/python/SpaGCN-master/SpaGCN_package')
import SpaGCN as spg
from scipy.sparse import issparse
import random, torch
import warnings
warnings.filterwarnings("ignore")
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import SpaGCN as spg
#In order to read in image data, we need to install some package. Here we recommend package "opencv"
#inatll opencv in python
#!pip3 install opencv-python
import cv2
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score, homogeneity_score

result = {}
def main(name):
    from scanpy import read_10x_h5
    path = '/home/guo/jiangtao/data/DLPFC/'
    path = '/home/cavin/jt/spatial_data/stDCL/DLPFC/'
    adata_sc = sc.read_visium(path + name, count_file='filtered_feature_bc_matrix.h5')
    adata = read_10x_h5(path+name+"/filtered_feature_bc_matrix.h5")
    adata.uns['spatial'] = adata_sc.uns['spatial']
    adata.obsm['spatial'] = adata_sc.obsm['spatial']
    spatial=pd.read_csv(path+name+"/spatial/tissue_positions_list.csv",sep=",",header=None,na_filter=False,index_col=0) 
    adata.obs["x1"]=spatial[1]
    adata.obs["x2"]=spatial[2]
    adata.obs["x3"]=spatial[3]
    adata.obs["x4"]=spatial[4]
    adata.obs["x5"]=spatial[5]
    adata.obs["x_array"]=adata.obs["x2"]
    adata.obs["y_array"]=adata.obs["x3"]
    adata.obs["x_pixel"]=adata.obs["x4"]
    adata.obs["y_pixel"]=adata.obs["x5"]

    adata=adata[adata.obs["x1"]==1]
    adata.var_names=[i.upper() for i in list(adata.var_names)]
    adata.var["genename"]=adata.var.index.astype("str")
    # adata.write_h5ad(path+name+"/sample_data.h5ad")
    ground_truth_df = pd.read_csv(path + name + '_truth.txt', delimiter='\t', header=None, dtype=str)
    adata.obs['ground_truth'] = ground_truth_df.iloc[:, 1].values  
    print(adata.uns['spatial'][name]['scalefactors']['tissue_hires_scalef'])
    scale = adata.uns['spatial'][name]['scalefactors']['tissue_hires_scalef']
    x_s,y_s = adata.uns['spatial'][name]['images']['hires'].shape[:2]
    x_s = int(x_s / scale)
    y_s = int(y_s / scale)

    print(adata.uns['spatial'][name]['images']['hires'].shape)
    img=cv2.imread(path+name+"/spatial/"+'tissue_hires_image.png')

    # img.resize(x_s,y_s)
    img = cv2.resize(img, (x_s, y_s)) 
    #Set coordinates
    x_array=adata.obs["x_array"].tolist()
    y_array=adata.obs["y_array"].tolist()
    x_pixel=adata.obs["x_pixel"].tolist()
    y_pixel=adata.obs["y_pixel"].tolist()

    #Test coordinates on the image
    img_new=img.copy()
    for i in range(len(x_pixel)):
        x=x_pixel[i]
        y=y_pixel[i]
        img_new[int(x-20):int(x+20), int(y-20):int(y+20),:]=0

    # cv2.imwrite('./sample_results/151673_map.jpg', img_new)

    #Calculate adjacent matrix
    s=1
    b=49
    adj=spg.calculate_adj_matrix(x=x_pixel,y=y_pixel, x_pixel=x_pixel, y_pixel=y_pixel, image=img, beta=b, alpha=s, histology=True)
    #If histlogy image is not available, SpaGCN can calculate the adjacent matrix using the fnction below
    #adj=calculate_adj_matrix(x=x_pixel,y=y_pixel, histology=False)
    # np.savetxt('./data/adj.csv', adj, delimiter=',')

    # adata=sc.read("./data/sample_data.h5ad")
    # adj=np.loadtxt('./data/adj.csv', delimiter=',')
    adata.var_names_make_unique()
    spg.prefilter_genes(adata,min_cells=3) # avoiding all genes are zeros
    spg.prefilter_specialgenes(adata)
    #Normalize and take log for UMI
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    p=0.5 
    #Find the l value given p
    l=spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

    #For this toy data, we set the number of clusters=7 since this tissue has 7 layers
    n_clusters=7
    #Set seed
    r_seed=t_seed=n_seed=100
    #Search for suitable resolution
    res=spg.search_res(adata, adj, l, n_clusters, start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)

    clf=spg.SpaGCN()
    clf.set_l(l)
    #Set seed
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)
    #Run
    cluster_num = 5 if name in ['151669', '151670', '151671', '151672'] else 7
    clf.train(adata,adj,init_spa=True,init="louvain",res=res, tol=5e-3, lr=0.05, max_epochs=200,n_clusters=cluster_num)
    y_pred, prob=clf.predict()
    adata.obs["pred"]= y_pred
    adata.obs["pred"]=adata.obs["pred"].astype('category')
    #Do cluster refinement(optional)
    #shape="hexagon" for Visium data, "square" for ST data.
    adj_2d=spg.calculate_adj_matrix(x=x_array,y=y_array, histology=False)
    refined_pred=spg.refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(), dis=adj_2d, shape="hexagon")
    adata.obs["refined_pred"]=refined_pred
    adata.obs["refined_pred"]=adata.obs["refined_pred"].astype('category')
    #Save results
    # adata.write_h5ad("./sample_results/results.h5ad")
    z,q=clf.model.predict(clf.embed,clf.adj_exp)
    z = z.detach().cpu().numpy()
    adata.obsm['emb'] = z

    obs_df = adata.obs.dropna()
    NMI_score = normalized_mutual_info_score(obs_df['pred'], obs_df['ground_truth'], average_method='max')
    HS_score = homogeneity_score(obs_df['pred'], obs_df['ground_truth'])
    adata.obs['domain'] = adata.obs['pred'].copy()
    # new_type = refine_label(adata, radius=10, key='domain')
    # adata.obs['domain'] = new_type
    filtered_domain = adata.obs['domain'][obs_df.index]  
    filtered_ground_truth = obs_df['ground_truth']
    assert len(filtered_domain) == len(
        filtered_ground_truth), f"Shape mismatch: domain has {len(filtered_domain)} elements, ground_truth has {len(filtered_ground_truth)} elements"
    ARI_score = adjusted_rand_score(filtered_domain, filtered_ground_truth)
    print('current slice:', name)
    print('History max ari:', ari[name])
    print('ARI:', ARI_score)
    print('NMI:', NMI_score)
    print('HS:', HS_score)
    r = {'ARI':ARI_score,'NMI':NMI_score,'HS':HS_score}
    result[name] = r
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
    adata.uns['ground_truth_colors'] = rainbow_hex[:cluster_num]
    adata.uns['domain_colors'] = rainbow_hex[:cluster_num]
    plt.rcParams.update({'axes.titlesize': 20})
    sc.pl.spatial(adata, color=["ground_truth", "domain"], title=['ground truth', 'SpaGCN (ARI=%.2f)' % ARI_score],show=False)
    file_dir = os.path.dirname(__file__)
    # dir = trainer.wavelet+'_'+str(trainer.level)
    dir = os.path.dirname(__file__)
    # os.makedirs(file_dir+'/DLPFC_final/'+dir, exist_ok=True)
    # plt.savefig(file_dir+'/DLPFC_final/'+dir+'/'+name+'.png', bbox_inches='tight', dpi=300)
    plt.savefig(dir+'/images/'+name+'.svg', bbox_inches='tight', dpi=300)
    plt.close()  

    # Plot UMAP
    sc.pp.neighbors(adata, use_rep='emb')
    sc.tl.umap(adata)
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.umap(adata, color=["domain", "ground_truth"], title=['SpaGCN (ARI=%.2f)'%ARI_score, "Ground Truth"],palette=rainbow_hex[:cluster_num],show=False)
    plt.savefig(dir+'/images/' + name + '_umap.svg', dpi=300, bbox_inches='tight')
    plt.close()
    # paga
    sc.tl.paga(adata, groups='ground_truth')
    plt.rcParams["figure.figsize"] = (4,3)
    sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20,palette=rainbow_hex[:cluster_num], 
                    title=name+'_SpaGCN', legend_fontoutline=2, show=False)
    plt.savefig(dir+'/images/' + name + '_paga.svg', dpi=300, bbox_inches='tight')
    plt.close()
    label_df = adata.obs['domain']
    label_df.to_csv(dir+'/images/' + name + '_label.csv')










if __name__ == '__main__':
    names = ['151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676']
    ari = {}
    ari = {'151507':0.56,
           '151508':0.52,
           '151509':0.52,
           '151510':0.55,
           '151669':0.66,
           '151670':0.65,
           '151671':0.79,
           '151672':0.76,
           '151673':0.62,
           '151674':0.65,
           '151675':0.67,
           '151676':0.59}
    #names = ['151510','151669','151670','151671','151672','151673','151674','151675','151676']
    # names = ['151675']
    
    for name in names:
        main(name)
    df = pd.DataFrame(result)
    df.to_csv(os.path.dirname(__file__)+'/images/metric.csv')
