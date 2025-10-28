# /home/guo/jiangtao/python/project2/helper.py
import sys
import os


sys.path.append("/home/cavin/jt/python/spaVAE-main/src/spaVAE")
# sys.path.append("/home/guo/jiangtao/python/spaVAE-main/src/spaVAE")
from spaVAE import SPAVAE
from preprocess import normalize, geneSelection
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score, homogeneity_score

import math
from time import time
import matplotlib.pyplot as plt
import torch

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import h5py
import scanpy as sc
import scipy.sparse as sp
print('import finished')

'''
Parameter setting
'''

class Args(object):
    def __init__(self):
        self.data_file = 'sample_151673.h5'
        self.select_genes = 0
        self.batch_size = "auto"
        self.maxiter = 5000
        self.train_size = 0.95
        self.patience = 200
        self.mask_prob = 0.1    # set mask probability, which means masking 0.1 random spots
        self.lr = 1e-3
        self.weight_decay = 1e-6
        self.noise = 0
        self.dropoutE = 0
        self.dropoutD = 0
        self.encoder_layers = [128, 64]
        self.GP_dim = 2
        self.Normal_dim = 8
        self.decoder_layers = [128]
        self.init_beta = 10
        self.min_beta = 4
        self.max_beta = 25
        self.KL_loss = 0.025
        self.num_samples = 1
        self.fix_inducing_points = True
        self.grid_inducing_points = True
        self.inducing_point_steps = 6
        self.inducing_point_nums = None
        self.fixed_gp_params = False
        self.loc_range = 20.
        self.kernel_scale = 20.
        # self.model_file = "model.pt"
        self.model_file = 'model'
        self.train_final_latent_file = "train_final_latent.txt"
        self.train_denoised_counts_file = "train_denoised_counts.txt"
        self.test_final_latent_file = "test_final_latent.txt"
        self.test_denoised_counts_file = "test_denoised_counts.txt"
        self.num_denoise_samples = 10000
        self.device = "cpu"

args = Args()

result = {}
def main(name):
    # data_mat = h5py.File(args.data_file, 'r')
    # x = np.array(data_mat['X']).astype('float64')
    # loc = np.array(data_mat['pos']).astype('float64')
    # data_mat.close()

    path = '/home/cavin/jt/spatial_data/stDCL/DLPFC/'
    # path = '/home/guo/jiangtao/data/DLPFC/'
    adata = sc.read_visium(path + name, count_file='filtered_feature_bc_matrix.h5')
    print("adata:", adata)
    import pandas as pd
    ground_truth_df = pd.read_csv(path + name + '_truth.txt', delimiter='\t', header=None, dtype=str)
    adata.obs['ground_truth'] = ground_truth_df.iloc[:, 1].values  
    # Data preprocessing
    adata.var_names_make_unique()
    x = np.array(adata.X.toarray()).astype('float64')
    loc = np.array(adata.obsm['spatial']).astype('float64')

    if args.batch_size == "auto":
        if x.shape[0] <= 1024:
            args.batch_size = 128
        elif x.shape[0] <= 2048:
            args.batch_size = 256
        else:
            args.batch_size = 512
    else:
        args.batch_size = int(args.batch_size)

    print(args)


    if args.select_genes > 0:
        importantGenes = geneSelection(x, n=args.select_genes, plot=False)
        x = x[:, importantGenes]
        # np.savetxt("selected_genes.txt", importantGenes, delimiter=",", fmt="%i")

    scaler = MinMaxScaler()
    loc = scaler.fit_transform(loc) * 20.

    print(x.shape)
    print(loc.shape)

    eps = 1e-5
    initial_inducing_points = np.mgrid[0:(1+eps):(1./args.inducing_point_steps), 0:(1+eps):(1./args.inducing_point_steps)].reshape(2, -1).T * 20.
    print(initial_inducing_points.shape)

    adata = normalize(adata,
                  size_factors=True,
                  normalize_input=True,
                  logtrans_input=True)

    sample_idx = np.arange(x.shape[0])
    np.random.shuffle(sample_idx)
    train_idx, test_idx = sample_idx[int(args.mask_prob*x.shape[0]):], sample_idx[:int(args.mask_prob*x.shape[0])]
    # np.savetxt(args.data_file[:-3]+"_train_index.txt", train_idx, delimiter=",", fmt="%i")
    # np.savetxt(args.data_file[:-3]+"_test_index.txt", test_idx, delimiter=",", fmt="%i")
    x_train, x_test = x[train_idx], x[test_idx]
    loc_train, loc_test = loc[train_idx], loc[test_idx]
    print(x_train.shape, x_test.shape)
    print(loc_train.shape, loc_test.shape)

    adata_train = sc.AnnData(x_train, dtype="float64")

    adata_train = normalize(adata_train,
                    size_factors=True,
                    normalize_input=True,
                    logtrans_input=True)

    adata_test = sc.AnnData(x_test)

    model = SPAVAE(input_dim=adata_train.n_vars, GP_dim=args.GP_dim, Normal_dim=args.Normal_dim, encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
        noise=args.noise, encoder_dropout=args.dropoutE, decoder_dropout=args.dropoutD,
        fixed_inducing_points=args.fix_inducing_points, initial_inducing_points=initial_inducing_points, 
        fixed_gp_params=args.fixed_gp_params, kernel_scale=args.kernel_scale, N_train=adata_train.n_obs, KL_loss=args.KL_loss, init_beta=args.init_beta, min_beta=args.min_beta, 
        max_beta=args.max_beta, dtype=torch.float64, device=args.device,dynamicVAE=False)

    print(str(model))


    # if not os.path.isfile(args.model_file):
    t0 = time()
    model.train_model(pos=loc_train, ncounts=adata_train.X, raw_counts=adata_train.raw.X, size_factors=adata_train.obs.size_factors,
                lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, num_samples=args.num_samples,
                train_size=args.train_size, maxiter=args.maxiter, patience=args.patience, save_model=False, model_weights=args.model_file)
    print('Training time: %d seconds.' % int(time() - t0))
    # else:
    #     model.load_model(args.model_file)

    final_latent = model.batching_latent_samples(X=loc_train, Y=adata_train.X, batch_size=args.batch_size)
    # np.savetxt(args.train_final_latent_file, final_latent, delimiter=",")
    print("Final latent shape:", final_latent.shape)

    # denoised_counts = model.batching_denoise_counts(X=loc_train, Y=adata_train.X, batch_size=args.batch_size, n_samples=25)
    # np.savetxt(args.train_denoised_counts_file, denoised_counts, delimiter=",")

    test_latent, test_denoised_counts = model.batching_predict_samples(X_test=loc_test, X_train=loc_train, Y_train=adata_train.X, batch_size=args.batch_size, n_samples=25)
    # np.savetxt(args.test_final_latent_file, test_latent, delimiter=",")
    # np.savetxt(args.test_denoised_counts_file, test_denoised_counts, delimiter=",")

    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn import metrics
    from sklearn.metrics import pairwise_distances

    # data_mat = h5py.File(args.data_file, 'r')
    # pos = np.array(data_mat['pos']).astype('float64')
    # y = np.array(data_mat['Y']).astype('U26') # ground-truth labels
    # data_mat.close()

    pos = np.array(adata.obsm['spatial']).astype('float64')
    y = np.array(adata.obs['ground_truth']).astype('U26') # ground-truth labels

    index = y!='NA'
    cluster_num = 5 if name in ['151669', '151670', '151671', '151672'] else 7
    # pred = KMeans(n_clusters=cluster_num, n_init=100).fit_predict(final_latent)
    # # np.savetxt("clustering_labels.txt", pred, delimiter=",", fmt="%i")

    # nmi = np.round(metrics.normalized_mutual_info_score(y[index], pred), 8)
    # ari = np.round(metrics.adjusted_rand_score(y[index], pred), 8)
    # print("NMI:", nmi, "; ARI:", ari)

    # dis = pairwise_distances(pos, metric="euclidean", n_jobs=-1).astype(np.double)
    # pred_refined = refine(np.arange(pred.shape[0]), pred, dis, shape="hexagon")
    # np.savetxt("refined_clustering_labels.txt", pred_refined, delimiter=",", fmt="%i")

    # nmi = np.round(metrics.normalized_mutual_info_score(y[index], pred_refined[index]), 8)
    # ari = np.round(metrics.adjusted_rand_score(y[index], pred_refined[index]), 8)
    # print("Refined NMI:", nmi, "; refined ARI:", ari)
    train_emb = final_latent
    test_emb = test_latent
    combined_latent = np.zeros((adata.X.shape[0], train_emb.shape[1]))
    combined_latent[train_idx] = train_emb
    combined_latent[test_idx] = test_emb

    adata.obsm['emb'] = combined_latent
    pred = KMeans(n_clusters=cluster_num, n_init=100).fit_predict(combined_latent)
    adata.obs['domain'] = pred
    adata.obs['domain'] = adata.obs['domain'].astype('int')
    adata.obs['domain'] = adata.obs['domain'].astype('category')

    obs_df = adata.obs.dropna()
    NMI_score = normalized_mutual_info_score(obs_df['domain'], obs_df['ground_truth'], average_method='max')
    HS_score = homogeneity_score(obs_df['domain'], obs_df['ground_truth'])
    adata.obs['domain'] = adata.obs['domain'].copy()
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
    sc.pl.spatial(adata, color=["ground_truth", "domain"], title=['ground truth', 'spaVAE (ARI=%.2f)' % ARI_score],show=False)
    file_dir = os.path.dirname(__file__)
    # dir = trainer.wavelet+'_'+str(trainer.level)
    dir = '/home/guo/jiangtao/python/wave/DLPFC/_spaVAE/images'
    # os.makedirs(file_dir+'/DLPFC_final/'+dir, exist_ok=True)
    # plt.savefig(file_dir+'/DLPFC_final/'+dir+'/'+name+'.png', bbox_inches='tight', dpi=300)
    plt.savefig(file_dir+'/images/'+name+'.svg', bbox_inches='tight', dpi=300)
    plt.close()  

    # Plot UMAP
    sc.pp.neighbors(adata, use_rep='emb')
    sc.tl.umap(adata)
    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.umap(adata, color=["domain", "ground_truth"], title=['spaVAE (ARI=%.2f)'%ARI_score, "Ground Truth"],palette=rainbow_hex[:cluster_num],show=False)
    plt.savefig(file_dir+'/images/' + name + '_umap.svg', dpi=300, bbox_inches='tight')
    plt.close()
    # paga
    sc.tl.paga(adata, groups='ground_truth')
    plt.rcParams["figure.figsize"] = (4,3)
    sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20,palette=rainbow_hex[:cluster_num], 
                    title=name+'_spaVAE', legend_fontoutline=2, show=False)
    plt.savefig(file_dir+'/images/' + name + '_paga.svg', dpi=300, bbox_inches='tight')
    plt.close()
    label_df = adata.obs['domain']
    label_df.to_csv(file_dir+'/images/' + name + '_label.csv')








if __name__ == '__main__':
    from tqdm import tqdm
    import pandas as pd
    names = ['151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676']
    ari = {}
    ari = {'151507':0.5,
           '151508':0.5,
           '151509':0.5,
           '151510':0.5,
           '151669':0.6,
           '151670':0.6,
           '151671':0.7,
           '151672':0.7,
           '151673':0.6,
           '151674':0.6,
           '151675':0.6,
           '151676':0.5}
    #names = ['151510','151669','151670','151671','151672','151673','151674','151675','151676']
    # names = ['151675']
    
    for name in tqdm(names):
        main(name)
    df = pd.DataFrame(result)
    df.to_csv(os.path.dirname(__file__)+'/images/metric.csv')

        
