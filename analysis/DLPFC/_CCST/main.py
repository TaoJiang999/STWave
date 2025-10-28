def main():
    import os
    os.environ['R_HOME'] = '/home/cavin/anaconda3/envs/tao/lib/R'
    import sys
    sys.path.append("/home/cavin/jt/python/CCST")
    import matplotlib
    matplotlib.use('Agg')
    #matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    #import pylab as pl
    #from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    from sklearn import metrics
    from sklearn.metrics import adjusted_rand_score
    from scipy import sparse
    #from sklearn.metrics import roc_curve, auc, roc_auc_score
    # from st_loading_utils import load_mPFC, load_mHypothalamus, load_her2_tumor, load_mMAMP, load_DLPFC, load_BC, load_mVC
    import numpy as np
    import pickle
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, ChebConv, GATConv, DeepGraphInfomax, global_mean_pool, global_max_pool  # noqa
    from torch_geometric.data import Data, DataLoader
    from datetime import datetime
    import argparse
    import scanpy as sc


    # from data_generation_ST import main
    # 
    # setting_combinations = [[7, '151507'], [7, '151508'], [7, '151509'], [7, '151510'], [5, '151669'], [5, '151670'], [5, '151671'], [5, '151672'], [7, '151673'], [7, '151674'], [7, '151675'], [7, '151676']]
    # for setting_combi in setting_combinations:
    #     data_name = setting_combi[1]  # '151673'
    #     p = argparse.ArgumentParser()
    #     p.add_argument( '--min_cells', type=float, default=5, help='Lowly expressed genes which appear in fewer than this number of cells will be filtered out')
    #     p.add_argument( '--Dim_PCA', type=int, default=200, help='The output dimention of PCA')
    #     # parser.add_argument( '--data_path', type=str, default='dataset/', help='The path to dataset')
    #     # parser.add_argument( '--data_name', type=str, default='V1_Breast_Cancer_Block_A_Section_1', help='The name of dataset')
    #     # parser.add_argument( '--generated_data_path', type=str, default='generated_data/', help='The folder to store the generated data')
    #     p.add_argument( '--data_path', type=str, default='/home/cavin/jt/spatial_data/stDCL/DLPFC/', help='The path to dataset')
    #     p.add_argument( '--data_name', type=str, default=data_name, help='The name of dataset')
    #     p.add_argument( '--generated_data_path', type=str, default='/home/cavin/jt/python/wave/DLPFC/_CCST/CCST/', help='The folder to store the generated data')
    #     a = p.parse_args() 

    #     main(a)

    parser = argparse.ArgumentParser()
    # ================Specify data type firstly===============
    parser.add_argument( '--data_type', default='nsc', help='"sc" or "nsc", \
    refers to single cell resolution datasets(e.g. MERFISH) and \
    non single cell resolution data(e.g. ST) respectively')
    # =========================== args ===============================
    parser.add_argument( '--data_name', type=str, default='V1_Breast_Cancer_Block_A_Section_1', help="'MERFISH' or 'V1_Breast_Cancer_Block_A_Section_1")
    parser.add_argument( '--lambda_I', type=float, default=0.3) #0.8 on MERFISH, 0.3 on ST
    parser.add_argument( '--data_path', type=str, default='/home/cavin/jt/python/wave/DLPFC/_CCST/CCST/', help='data path')
    parser.add_argument( '--model_path', type=str, default='model')
    parser.add_argument( '--embedding_data_path', type=str, default='Embedding_data')
    parser.add_argument( '--result_path', type=str, default='results')
    parser.add_argument( '--DGI', type=int, default=1, help='run Deep Graph Infomax(DGI) model, otherwise direct load embeddings')
    parser.add_argument( '--load', type=int, default=0, help='Load pretrained DGI model')
    parser.add_argument( '--num_epoch', type=int, default=5000, help='numebr of epoch in training DGI')
    parser.add_argument( '--hidden', type=int, default=256, help='hidden channels in DGI')
    parser.add_argument( '--PCA', type=int, default=1, help='run PCA or not')
    parser.add_argument( '--cluster', type=int, default=1, help='run cluster or not')
    parser.add_argument( '--n_clusters', type=int, default=5, help='number of clusters in Kmeans, when ground truth label is not avalible.') #5 on MERFISH, 20 on Breast
    parser.add_argument( '--draw_map', type=int, default=0, help='run drawing map')
    parser.add_argument( '--diff_gene', type=int, default=0, help='Run differential gene expression analysis')
    parser.add_argument( '--batch_size', type=int, default=512, help='training batch size')
    parser.add_argument( '--gpu_id', type=str, default="2", help='default gpu id')

    args = parser.parse_args()
    iters=1 # for script testing
    # iters = 20 # for boxplotting
    args.embedding_data_path = os.path.dirname(__file__)+ '/CCST/'


    """DLPFC"""
    setting_combinations = [[7, '151507'], [7, '151508'], [7, '151509'], [7, '151510'], [5, '151669'], [5, '151670'], [5, '151671'], [5, '151672'], [7, '151673'], [7, '151674'], [7, '151675'], [7, '151676']]
    # setting_combinations = [[5, '151669'], [5, '151670'], [5, '151671'], [5, '151672'], [7, '151673'], [7, '151674'], [7, '151675'], [7, '151676']]
    # setting_combinations = [[5, '151671'], [5, '151672'], [7, '151673'], [7, '151674'], [7, '151675'], [7, '151676']]
    # setting_combinations = [[7, '151673'], [7, '151674'], [7, '151675'], [7, '151676']]
    for setting_combi in setting_combinations:

        args.n_clusters = setting_combi[0]  # 7

        args.data_name = setting_combi[1]  # '151673'
        dataset = setting_combi[1]
        args.data_type = 'nsc'
        dir = '/home/cavin/jt/spatial_data/stDCL/DLPFC/'
        #    ad = load_DLPFC(root_dir=dir_, section_id=args.data_name)
        ad = sc.read_visium(os.path.join(dir,args.data_name))
        import pandas as pd
        ground_truth_df = pd.read_csv(dir + args.data_name + '_truth.txt', delimiter='\t', header=None, dtype=str)
        ad.obs['ground_truth'] = ground_truth_df.iloc[:, 1].values  
        # Data preprocessing
        ad.var_names_make_unique()

        aris = []
        args.embedding_data_path = os.path.dirname(__file__)+ '/CCST' +'/'+ args.data_name +'/'
        args.model_path = os.path.dirname(__file__)+ '/CCST/model/' + args.data_name +'/'
        args.result_path = os.path.dirname(__file__)+ '/CCST/result/' + args.data_name +'/'
        if not os.path.exists(args.embedding_data_path):
            os.makedirs(args.embedding_data_path)
        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)
        args.result_path = args.result_path+'lambdaI'+str(args.lambda_I) +'/'
        if not os.path.exists(args.result_path):
            os.makedirs(args.result_path)



        print ('------------------------Model and Training Details--------------------------')
        print(args)

        for iter_ in range(iters):


            if args.data_type == 'sc': # should input a single cell resolution dataset, e.g. MERFISH
                from CCST_merfish_utils import CCST_on_MERFISH
                CCST_on_MERFISH(args)
            elif args.data_type == 'nsc': # should input a non-single cell resolution dataset, e.g. V1_Breast_Cancer_Block_A_Section_1
                from CCST_ST_utils import CCST_on_ST
                preds = CCST_on_ST(args)
            else:
                print('Data type not specified')

            # calculate metric ARI
            # obs_df = ad.obs.dropna()
            # print(preds)
            # print(obs_df['original_clusters'].to_list())
            # ARI = adjusted_rand_score(np.array(preds)[:, 1], obs_df['ground_truth'].to_list())

            # print('Dataset:', dataset)
            # print('ARI:', ARI)
            # aris.append(ARI)
        # print('Dataset:', dataset)
        # print(aris)
        # print(np.mean(aris))
        # with open('ccst_aris.txt', 'a+') as fp:
        #     fp.write('DLPFC' + dataset + ' ')
        #     fp.write(' '.join([str(i) for i in aris]))
        #     fp.write('\n')




if __name__ == '__main__':
    # main()
    # main()
    import os
    import scanpy as sc
    import numpy as np
    from matplotlib import pyplot as plt
    from sklearn.metrics import adjusted_rand_score
    from tqdm import tqdm
    from sklearn.metrics.cluster import normalized_mutual_info_score, homogeneity_score
    names = ['151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676']
    emb_dir = '/home/cavin/jt/python/wave/DLPFC/_CCST/CCST'
    emb_dir_list = []
    label_dir = '/home/cavin/jt/python/wave/DLPFC/_CCST/CCST/result'
    label_dir_list = []
    result = {}
    for name in names:
        # python/wave/DLPFC/_CCST/CCST/151507/lambdaI0.3_epoch5000_Embed_X.npy
        emb_dir_list.append(os.path.join(emb_dir,name,'lambdaI0.3_epoch5000_Embed_X.npy'))
        label_dir_list.append(os.path.join(label_dir,name,'lambdaI0.3','types.txt'))
    adata_dir = '/home/cavin/jt/spatial_data/stDCL/DLPFC/'
    for name,label_path,emb_path in zip(names, label_dir_list, emb_dir_list):
        adata = sc.read_visium(path=os.path.join(adata_dir, name))
        print("adata:", adata)
        import pandas as pd
        ground_truth_df = pd.read_csv(os.path.join(adata_dir, name+'_truth.txt'), delimiter='\t', header=None, dtype=str)
        adata.obs['ground_truth'] = ground_truth_df.iloc[:, 1].values  
        # Data preprocessing
        adata.var_names_make_unique()
        label = pd.read_csv(label_path, index_col=None,header=None,sep='\t')
        # emb = pd.read_csv(emb_path, index_col=0,header=0,sep=',')
        emb = np.load(emb_path,allow_pickle=False)
        print("Shape of embed_x:", emb.shape)
        print("Data type:", emb.dtype)
        print("First few rows:\n", emb[:5])

        cluster_num = 5 if name in ['151669', '151670', '151671', '151672'] else 7
        adata.obsm['emb'] = emb
        adata.obs['domain'] = label.values[:, 1]
        adata.obs['domain'] = adata.obs['domain'].astype('int')
        adata.obs['domain'] = adata.obs['domain'].astype('category')

        adata = adata[~adata.obs['ground_truth'].isnull()]
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
        tqdm.write('saving plot')
        adata.uns['ground_truth_colors'] = rainbow_hex[:cluster_num]
        adata.uns['domain_colors'] = rainbow_hex[:cluster_num]
        plt.rcParams.update({'axes.titlesize': 20})
        sc.pl.spatial(adata, color=["ground_truth", "domain"], title=['ground truth', 'CCST (ARI=%.2f)' % ARI_score],show=False)
        dir = os.path.dirname(__file__)
        os.makedirs(dir+'/images', exist_ok=True)
        
        plt.savefig(dir+'/images/'+name+'.svg')
        plt.close()  
        # Plot UMAP
        sc.pp.neighbors(adata, use_rep='emb')
        sc.tl.umap(adata)
        plt.rcParams["figure.figsize"] = (3, 3)
        sc.pl.umap(adata, color=["domain", "ground_truth"], title=['CCST (ARI=%.2f)'%ARI_score, "Ground Truth"],palette=rainbow_hex[:cluster_num],show=False)
        plt.savefig(dir+'/images/'+ name + '_umap.svg')
        plt.close()
        # paga
        sc.tl.paga(adata, groups='ground_truth')
        plt.rcParams["figure.figsize"] = (4,3)
        sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20,palette=rainbow_hex[:cluster_num], 
                        title=name+'_CCST', legend_fontoutline=2, show=False)
        plt.savefig(dir+'/images/' + name + '_paga.svg')
        plt.close()
        label_df = adata.obs['domain']
        label_df.to_csv(dir+'/images/' + name + '_label.csv')
    df = pd.DataFrame(result)
    df.to_csv(os.path.dirname(__file__)+'/images/metric.csv')
