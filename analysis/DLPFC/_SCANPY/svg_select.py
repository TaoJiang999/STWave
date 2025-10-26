import pandas as pd
import numpy as np
import warnings
import scipy.sparse as ss
import scanpy as sc
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
import itertools
import scipy.sparse as ss
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
import networkx as nx
warnings.filterwarnings("ignore")


def get_laplacian_mtx(adata,
                      num_neighbors=6,
                      spatial_key=['array_row', 'array_col'],
                      normalization=False):
    """
    Obtain the Laplacian matrix or normalized laplacian matrix.

    Parameters
    ----------
    adata : AnnData
        adata.X is the normalized count matrix. Besides, the spatial coordinat-
        es could be found in adata.obs or adata.obsm.
    num_neighbors: int, optional
        The number of neighbors for each node/spot/pixel when contrcut graph.
        The defalut if 6.
    spatial_key=None : list | string
        Get the coordinate information by adata.obsm[spaital_key] or
        adata.var[spatial_key]. The default is ['array_row', 'array_col'].
    normalization : bool, optional
        Whether need to normalize laplacian matrix. The default is False.

    Raises
    ------
    KeyError
        The coordinates should be found at adata.obs[spatial_names] or
        adata.obsm[spatial_key]

    Returns
    -------
    lap_mtx : csr_matrix
        The laplcaian matrix or mormalized laplcian matrix.

    """
    if spatial_key in adata.obsm_keys():
        adj_mtx = kneighbors_graph(adata.obsm[spatial_key],
                                   n_neighbors=num_neighbors)
    elif set(spatial_key) <= set(adata.obs_keys()):
        adj_mtx = kneighbors_graph(adata.obs[spatial_key],
                                   n_neighbors=num_neighbors)
    else:
        raise KeyError("%s is not avaliable in adata.obsm_keys" % \
                       spatial_key + " or adata.obs_keys")

    adj_mtx = nx.adjacency_matrix(nx.Graph(adj_mtx))
    # Degree matrix
    deg_mtx = adj_mtx.sum(axis=1)
    deg_mtx = create_degree_mtx(deg_mtx)
    # Laplacian matrix
    # Whether need to normalize laplcian matrix
    if not normalization:
        lap_mtx = deg_mtx - adj_mtx
    else:
        deg_mtx = np.array(adj_mtx.sum(axis=1)) ** (-0.5)
        deg_mtx = create_degree_mtx(deg_mtx)
        lap_mtx = ss.identity(deg_mtx.shape[0]) - deg_mtx @ adj_mtx @ deg_mtx

    return lap_mtx


def create_degree_mtx(diag):
    diag = np.array(diag)
    diag = diag.flatten()
    row_index = list(range(diag.size))
    col_index = row_index
    sparse_mtx = ss.coo_matrix((diag, (row_index, col_index)),
                               shape=(diag.size, diag.size))

    return sparse_mtx


def rank_gene_smooth(adata,
                     ratio_low_freq='infer',
                     ratio_high_freq='infer',
                     ratio_neighbors='infer',
                     spatial_info=['array_row', 'array_col'],
                     normalize_lap=False,
                     filter_peaks=True,
                     S=5,
                     cal_pval=True):
    """
    Rank genes to find spatially variable genes by graph Fourier transform.

    Parameters
    ----------
    adata : AnnData
        adata.X is the normalized count matrix. Besides, the spatial coordinat-
        es could be found in adata.obs or adata.obsm.
    ratio_low_freq : float | "infer", optional
        The ratio_low_freq will be used to determine the number of the FMs with
        low frequencies. Indeed, the ratio_low_freq * sqrt(number of spots) low
        frequecy FMs will be calculated. If 'infer', the ratio_low_freq will be
        set to 1.0. The default is 'infer'.
    ratio_high_freq: float | 'infer', optional
        The ratio_high_freq will be used to determine the number of the FMs of
        high frequencies. Indeed, the ratio_high_freq * sqrt(number of spots)
        high frequecy FMs will be calculated. If 'infer', the ratio_high_freq
        will be set to 1.0. The default is 'infer'.
    ratio_neighbors: float | 'infer', optional
        The ratio_neighbors will be used to determine the number of neighbors
        when contruct the KNN graph by spatial coordinates. Indeed, ratio_neig-
        hobrs * sqrt(number of spots) / 2 indicates the K. If 'infer', the para
        will be set to 1.0. The default is 'infer'.
    spatial_info : list | tupple | string, optional
        The column names of spaital coordinates in adata.obs_names or key
        in adata.varm_keys() to obtain spatial information. The default
        is ['array_row', 'array_col'].
    normalize_lap : bool, optional
        Whether need to normalize laplacian matrix. The default is false.
    filter_peaks: bool, optional
        For calculated vectors/signals in frequency/spectral domian, whether
        filter low peaks to stress the important peaks. The default is True.
    S: int, optional
        The sensitivity parameter in Kneedle algorithm. A large S will enable
        more genes indentified as SVGs according to gft_score. The default is
        5.
    cal_pval : bool, optional
        Whether need to calculate p val by mannwhitneyu. The default is False.
    Returns
    -------
    score_df : dataframe
        Return gene information.

    """
    # Ensure parameters
    if ratio_low_freq == 'infer':
        num_low_frequency = int(np.ceil(np.sqrt(adata.shape[0])))
    else:
        num_low_frequency = int(np.ceil(np.sqrt(adata.shape[0]) * \
                                        ratio_low_freq))
    if ratio_high_freq == 'infer':
        num_high_frequency = int(np.ceil(np.sqrt(adata.shape[0])))
    else:
        num_high_frequency = int(np.ceil(np.sqrt(adata.shape[0]) * \
                                         ratio_high_freq))

    if ratio_neighbors == 'infer':
        num_neighbors = int(np.ceil(np.sqrt(adata.shape[0]) / 2))
    else:
        num_neighbors = int(np.ceil(np.sqrt(adata.shape[0]) / 2 \
                                    * ratio_neighbors))
    if adata.shape[0] <= 500:
        num_neighbors = 4

    # Ensure gene index uniquely and all gene had expression
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=1)

    # *************** Construct graph and corresponding matrixs ***************
    lap_mtx = get_laplacian_mtx(adata, num_neighbors=num_neighbors,
                                spatial_key=spatial_info,
                                normalization=normalize_lap)

    # Next, calculate the eigenvalues and eigenvectors of the Laplace matrix
    # Fourier bases of low frequency
    eigvals_s, eigvecs_s = ss.linalg.eigsh(lap_mtx.astype(float),
                                           k=num_low_frequency,
                                           which='SM')
    if num_high_frequency > 0:
        # Fourier bases of high frequency
        eigvals_l, eigvecs_l = ss.linalg.eigsh(lap_mtx.astype(float),
                                               k=num_high_frequency,
                                               which='LM')
        eigvals = np.concatenate((eigvals_s, eigvals_l))  # eigenvalues
        eigvecs = np.concatenate((eigvecs_s, eigvecs_l), axis=1)  # eigenvectors
    else:
        eigvals = eigvals_s
        eigvecs = eigvecs_s

    # ************************ Graph Fourier Tranform *************************
    # Calculate GFT
    eigvecs_T = eigvecs.transpose()
    if type(adata.X) == np.ndarray:
        exp_mtx = preprocessing.scale(adata.X)
    else:
        exp_mtx = preprocessing.scale(adata.X.toarray())

    frequency_array = np.matmul(eigvecs_T, exp_mtx)
    frequency_array = np.abs(frequency_array)

    # Filter noise peaks
    if filter_peaks == True:
        frequency_array_thres_low = \
            np.quantile(frequency_array[:num_low_frequency, :],
                        q=0.5, axis=0)
        frequency_array_thres_high = \
            np.quantile(frequency_array[num_low_frequency:, :],
                        q=0.5, axis=0)
        for j in range(frequency_array.shape[1]):
            frequency_array[:num_low_frequency, :] \
                [frequency_array[:num_low_frequency, j] <= \
                 frequency_array_thres_low[j], j] = 0
            frequency_array[num_low_frequency:, :] \
                [frequency_array[num_low_frequency:, j] <= \
                 frequency_array_thres_high[j], j] = 0

    frequency_array = preprocessing.normalize(frequency_array,
                                              norm='l1',
                                              axis=0)

    eigvals = np.abs(eigvals)
    eigvals_power = np.exp(- eigvals) * 5
    score_list = np.matmul(eigvals_power, frequency_array)
    score_max = np.matmul(eigvals_power, (1 / len(eigvals)) * \
                          np.ones(len(eigvals)))
    score_list = score_list / score_max
    # print("Graph Fourier Transform finished!")

    # Rank genes according to smooth score
    adata.var["gft_score"] = score_list
    score_df = adata.var["gft_score"]
    score_df = pd.DataFrame(score_df)
    score_df = score_df.sort_values(by="gft_score", ascending=False)
    score_df.loc[:, "svg_rank"] = range(1, score_df.shape[0] + 1)
    adata.var["svg_rank"] = score_df.reindex(adata.var_names).loc[:, "svg_rank"]
    # print("SVG ranking could be found in adata.var['svg_rank']")

    # Determine cutoff of gft_score
    from kneed import KneeLocator
    magic = KneeLocator(score_df.svg_rank.values,
                        score_df.gft_score.values,
                        direction='decreasing',
                        curve='convex',
                        S=S)
    score_df['cutoff_gft_score'] = False
    score_df['cutoff_gft_score'][:(magic.elbow + 1)] = True
    adata.var['cutoff_gft_score'] = score_df['cutoff_gft_score']
    # print("""The spatially variable genes judged by gft_score could be found
    #       in adata.var['cutoff_gft_score']""")
    adata.varm['freq_domain_svg'] = frequency_array.transpose()
    # print("""Gene signals in frequency domain when detect SVGs could be found
    #       in adata.varm['freq_domain_svg']""")
    adata.uns['frequencies_svg'] = eigvals
    adata.uns['fms_low'] = eigvecs_s
    adata.uns['fms_high'] = eigvecs_l

    # ****************** calculate pval ***************************
    if cal_pval == True:
        pval_list = _test_significant_freq(
            freq_array=adata.varm['freq_domain_svg'],
            cutoff=num_low_frequency)
        from statsmodels.stats.multitest import multipletests
        qval_list = multipletests(np.array(pval_list), method='fdr_by')[1]
        adata.var['pvalue'] = pval_list
        adata.var['qvalue'] = qval_list
        score_df = adata.var.loc[score_df.index, :].copy()

    return score_df

def _test_significant_freq(freq_array,
                           cutoff,
                           num_pool=200):
    """
    Significance test by camparing the intensities in low frequency FMs and
    in high frequency FMs.

    Parameters
    ----------
    freq_array : array
        The graph signals of genes in frequency domain.
    cutoff : int
        Watershed between low frequency signals and high frequency signals.
    num_pool : int, optional
        The cores used for umltiprocess calculation to accelerate speed. The
        default is 200.

    Returns
    -------
    array
        The calculated p values.

    """
    from scipy.stats import wilcoxon, mannwhitneyu, ranksums, combine_pvalues
    from multiprocessing.dummy import Pool as ThreadPool

    def _test_by_feq(gene_index):
        freq_signal = freq_array[gene_index, :]
        freq_1 = freq_signal[:cutoff]
        freq_1 = freq_1[freq_1 > 0]
        freq_2 = freq_signal[cutoff:]
        freq_2 = freq_2[freq_2 > 0]
        if freq_1.size <= 50 or freq_2.size <= 50:
            freq_1 = np.concatenate((freq_1, freq_1, freq_1, freq_1))
            freq_2 = np.concatenate((freq_2, freq_2, freq_2, freq_2))
        pval = ranksums(freq_1, freq_2, alternative='greater').pvalue
        return pval

    gene_index_list = list(range(freq_array.shape[0]))
    pool = ThreadPool(num_pool)
    res = pool.map(_test_by_feq, gene_index_list)

    return res



def low_pass_enhancement(adata,
                         ratio_low_freq='infer',
                         ratio_high_freq='infer',
                         ratio_neighbors='infer',
                         c=0.0001,
                         spatial_info=['array_row', 'array_col'],
                         normalize_lap=False,
                         inplace=False):
    """
    Implement gene expression with low-pass filter. After this step, the
    spatially variables genes will be more smooth than previous. The function
    can also be treated as denoising. Note that the denosing results is related
    to spatial graph topology so that only the resulsts of spatially variable
    genes could be convincing.

    Parameters
    ----------
    adata : AnnData
        adata.X is the normalized count matrix. Besides, the spatial coordinat-
        es of all spots should be found in adata.obs or adata.obsm.
    ratio_low_freq : float | "infer", optional
        The ratio_low_freq will be used to determine the number of the FMs with
        low frequencies. Indeed, the ratio_low_freq * sqrt(number of spots) low
        frequecy FMs will be calculated. The default is 'infer'.
    ratio_high_freq: float | 'infer', optional
        The ratio_high_freq will be used to determine the number of the FMs with
        high frequencies. Indeed, the ratio_high_freq * sqrt(number of spots)
        high frequecy FMs will be calculated. If 'infer', the ratio_high_freq
        will be set to 0. The default is 'infer'.
        A high can achieve better smothness. c should be setted to [0, 0.1].
    ratio_neighbors: float | 'infer', optional
        The ratio_neighbors will be used to determine the number of neighbors
        when contruct the KNN graph by spatial coordinates. Indeed, ratio_neig-
        hobrs * sqrt(number of spots) / 2 indicates the K. If 'infer', the para
        will be set to 1.0. The default is 'infer'.
    c: float, optional
        c balances the smoothness and difference with previous expresssion.
    spatial_info : list or tupple, optional
        The column names of spaital coordinates in adata.obs_names or key
        in adata.obsm_keys() to obtain spatial information. The default
        is ['array_row', 'array_col'].
    normalize_lap : bool. optional
        Whether need to normalize the Laplcian matrix. The default is False.
    inplace: bool, optional


    Returns
    -------
    count_matrix: DataFrame

    """
    import scipy.sparse as ss
    if ratio_low_freq == 'infer':
        if adata.shape[0] <= 800:
            num_low_frequency = min(15 * int(np.ceil(np.sqrt(adata.shape[0]))),
                                    adata.shape[0])
        elif adata.shape[0] <= 2000:
            num_low_frequency = 12 * int(np.ceil(np.sqrt(adata.shape[0])))
        elif adata.shape[0] <= 10000:
            num_low_frequency = 10 * int(np.ceil(np.sqrt(adata.shape[0])))
        else:
            num_low_frequency = 4 * int(np.ceil(np.sqrt(adata.shape[0])))
    else:
        num_low_frequency = int(np.ceil(np.sqrt(adata.shape[0]) * \
                                        ratio_low_freq))
    if ratio_high_freq == 'infer':
        num_high_frequency = 0 * int(np.ceil(np.sqrt(adata.shape[0])))
    else:
        num_high_frequency = int(np.ceil(np.sqrt(adata.shape[0]) * \
                                         ratio_high_freq))

    if ratio_neighbors == 'infer':
        if adata.shape[0] <= 500:
            num_neighbors = 4
        else:
            num_neighbors = int(np.ceil(np.sqrt(adata.shape[0]) / 2))
    else:
        num_neighbors = int(np.ceil(np.sqrt(adata.shape[0]) / 2 \
                                    * ratio_neighbors))

    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=1)
    # Get Laplacian matrix according to coordinates
    lap_mtx = get_laplacian_mtx(adata,
                                num_neighbors=num_neighbors,
                                spatial_key=spatial_info,
                                normalization=normalize_lap)

    # Fourier modes of low frequency
    eigvals_s, eigvecs_s = ss.linalg.eigsh(lap_mtx.astype(float),
                                           k=num_low_frequency,
                                           which='SM')
    if num_high_frequency > 0:
        # Fourier modes of high frequency
        eigvals_l, eigvecs_l = ss.linalg.eigsh(lap_mtx.astype(float),
                                               k=num_high_frequency,
                                               which='LM')
        eigvals = np.concatenate((eigvals_s, eigvals_l))  # eigenvalues
        eigvecs = np.concatenate((eigvecs_s, eigvecs_l), axis=1)  # eigenvectors
    else:
        eigvals = eigvals_s
        eigvecs = eigvecs_s

    # *********************** Graph Fourier Tranform **************************
    # Calculate GFT
    eigvecs_T = eigvecs.transpose()
    if not ss.issparse(adata.X):
        exp_mtx = adata.X
    else:
        exp_mtx = adata.X.toarray()
    frequency_array = np.matmul(eigvecs_T, exp_mtx)
    # low-pass filter
    filter_list = [1 / (1 + c * eigv) for eigv in eigvals]
    filter_array = np.matmul(np.diag(filter_list), frequency_array)
    filter_array = np.matmul(eigvecs, filter_array)
    filter_array[filter_array < 0] = 0

    # whether need to replace original count matrix
    if inplace and not ss.issparse(adata.X):
        adata.X = filter_array
    elif inplace:
        import scipy.sparse as ss
        adata.X = ss.csr.csr_matrix(filter_array)

    filter_array = pd.DataFrame(filter_array,
                                index=adata.obs_names,
                                columns=adata.var_names)
    pass

