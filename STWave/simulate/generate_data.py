import os
import numpy as np
import pandas as pd
import scanpy as sc
import os
from PIL import Image
import matplotlib.pyplot as plt
import sys
from .utils import simulate_ST


# simu = sys.argv[1]
# rep = sys.argv[2]
# nspots = int(sys.argv[3])
# disperse = sys.argv[4]
# simu = 6
# rep = 1
# nspots = 10000
# disperse = 0.1

def main(simu = 6,rep = 1,nspots = 10000,disperse = 0.1):
    
    res_dir = f'simu{simu}/rep{rep}'
    res_dir = '/home/guo/jt/data/simulate/'+res_dir
    if not os.path.exists(res_dir):
        os.makedirs(res_dir,exist_ok=True)

    # make spatial pattern

    image_path = f'simu{simu}.png'
    image_path = '/home/guo/jt/data/simulate/'+image_path
    target_width = target_height = round(np.sqrt(nspots))


    image = Image.open(image_path)
    image = image.resize((target_width, target_height), resample=Image.BICUBIC)


    data = []
    for y in range(target_height):
        for x in range(target_width):
            pixel_color = image.getpixel((x, y))
            data.append({
                'x': x,
                'y': y,
                'color': f'#{pixel_color[0]:02X}{pixel_color[1]:02X}{pixel_color[2]:02X}'
            })

    spatial_df = pd.DataFrame(data)
    mapping = dict(zip(spatial_df['color'].value_counts()[:9].index.to_list(),[str(i) for i in range(9)]))
    spatial_df['domain'] = spatial_df['color'].map(mapping)
    spatial_df['domain'] = spatial_df['domain'].fillna('0')
    spatial_df['domain'] = spatial_df['domain'].astype('str')
    print(spatial_df['domain'].head(20))
    unique_spatial_types = spatial_df['domain'].unique()
    print(unique_spatial_types)
    ######## simulate data
    # please download the The integrated Human Lung Cell Atlas (HLCA) v1.0 data using the URL https://datasets.cellxgene.cziscience.com/cd9ac225-682a-4f39-b0de-29caeb532bec.h5ad
    adata = sc.read_h5ad('/home/guo/jt/data/HLCA/HLCA.h5ad')
    used_types = ['Macrophages','T cell lineage','Secretory','AT2','Basal','Fibroblasts','Multiciliated lineage','Monocytes','B cell lineage',
                'EC capillary','Innate lymphoid cell NK','Dendritic cells']
    HLCA_sub = adata[adata.obs['ann_level_3'].isin(used_types)]
    adata = simulate_ST(HLCA_sub,spatial_df,disperse_frac=float(disperse))
    del HLCA_sub
    adata.obs = adata.obs.filter(['ann_level_3'])
    sc.pp.filter_genes(adata, min_cells=round(adata.shape[0]/10000))
    adata.write_h5ad(f'{res_dir}/data.h5ad',compression='gzip')

    # plot the ground truth label
    sc.pl.embedding(adata,color='ann_level_3',basis='spatial',s=6,show=False, palette='tab20')
    plt.savefig(f'{res_dir}/GT.pdf', bbox_inches='tight')

    ######## simulate sparse data (optional)
    preserve_p = 0.5
    select_idx = np.random.choice(np.arange(adata.shape[0]),round(adata.shape[0]*preserve_p))
    adata_sp = adata[select_idx]
    sc.pp.filter_genes(adata_sp, min_cells=round(adata_sp.shape[0]/10000))
    adata_sp.write_h5ad(f'{res_dir}/data_sp.h5ad',compression='gzip')
    del adata
    del adata_sp

if __name__ == '__main__':
    sims = [1,2,3,4,5,6]
    rep = 1
    nspots = [80000,160000,240000,320000,480000,640000]
    disperses = [0.1,0.1,0.2,0.2,0.3,0.3] 
    for sim,nspot,disperse in zip(sims,nspots,disperses):
        print(f'current param:simu={sim},rep={rep},nspots={nspot},disperse={disperse}')
        main(simu=sim,rep=rep,nspots=nspot,disperse=disperse)