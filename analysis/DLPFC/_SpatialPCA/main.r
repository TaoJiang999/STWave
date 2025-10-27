library(SpatialPCA)
library(ggplot2)


sample_names=c("151507", "151508", "151509", "151510", "151669", "151670", "151671" ,"151672","151673", "151674" ,"151675" ,"151676")
# i=9 # Here we take the 9th sample as example, in total there are 12 samples (numbered as 1-12), the user can test on other samples if needed.
clusterNum=c(7,7,7,7,5,5,5,5,7,7,7,7) # each sample has different ground truth cluster number
index = c(5,6,7,8,9,10,11,12)
for (i in index){
    load(paste0("/home/cavin/jiangtao/spatial_data/rdara/DLPFC/LIBD_sample",i,".RData")) 
    print(dim(count_sub)) # The count matrix
    print(dim(xy_coords)) # The x and y coordinates. We flipped the y axis for visualization

    cell_ids <- colnames(count_sub)
    # cell_ids <- rownames(xy_coords)
    print("cell_ids")
    # print(cell_ids)
    print(length(cell_ids))
    # location matrix: n x 2, count matrix: g x n.
    # here n is spot number, g is gene number.
    xy_coords = as.matrix(xy_coords)
    rownames(xy_coords) = colnames(count_sub) # the rownames of location should match with the colnames of count matrix
    LIBD = CreateSpatialPCAObject(counts=count_sub, location=xy_coords, project = "SpatialPCA",gene.type="spatial",sparkversion="spark",numCores_spark=5,gene.number=3000, customGenelist=NULL,min.loctions = 20, min.features=20)


    LIBD = SpatialPCA_buildKernel(LIBD, kerneltype="gaussian", bandwidthtype="SJ",bandwidth.set.by.user=NULL)
    LIBD = SpatialPCA_EstimateLoading(LIBD,fast=FALSE,SpatialPCnum=20) 
    LIBD = SpatialPCA_SpatialPCs(LIBD, fast=FALSE)
    print("keys")
    print(slotNames(LIBD))
    # print(LIBD)

    clusterlabel= walktrap_clustering(clusternum=clusterNum[i],latent_dat=LIBD@SpatialPCs,knearest=70 ) 
    # here for all 12 samples in LIBD, we set the same k nearest number in walktrap_clustering to be 70. 
    # for other Visium or ST data, the user can also set k nearest number as round(sqrt(dim(SpatialPCAobject@SpatialPCs)[2])) by default.
    clusterlabel_refine = refine_cluster_10x(clusterlabels=clusterlabel,location=LIBD@location,shape="hexagon")

    # cbp=c("#9C9EDE" ,"#5CB85C" ,"#E377C2", "#4DBBD5" ,"#FED439" ,"#FF9896", "#FFDC91")
    # plot_cluster(location=xy_coords,clusterlabel=clusterlabel_refine,pointsize=1.5,textsize=20 ,title_in=paste0("SpatialPCA"),color_in=cbp)


    # truth = KRM_manual_layers_sub$layer_guess_reordered[match(colnames(LIBD@normalized_expr),colnames(count_sub))]
    # cbp=c("#5CB85C" ,"#9C9EDE" ,"#FFDC91", "#4DBBD5" ,"#FF9896" ,"#FED439", "#E377C2", "#FED439")
    # plot_cluster(location=xy_coords,truth,pointsize=1.5,textsize=20 ,title_in=paste0("Ground truth"),color_in=cbp)

    set.seed(1234)
    # p_UMAP = plot_RGB_UMAP(LIBD@location,LIBD@SpatialPCs,pointsize=2,textsize=15)
    # p_UMAP$figure

    # p_tSNE = plot_RGB_tSNE(LIBD@location,LIBD@SpatialPCs,pointsize=2,textsize=15)
    # p_tSNE$figure


    
    pcs <- LIBD@SpatialPCs  

    print("clusterlabel length")
    print(length(clusterlabel))
    print("clusterlabel_refine length")
    print(length(clusterlabel_refine))
    print("pcs dim")
    print(dim(pcs))


    print("pcs dim")
    print(dim(pcs))
    pcs <- t(pcs)
    print("t pcs dim")
    print(dim(pcs))
    barcodes <- rownames(pcs)
    print("barcode dim")
    print(dim(barcodes))
    # cell_ids <- as.character(seq_len(nrow(LIBD@location)))
    cell_ids <- rownames(pcs)
    label_df <- data.frame(
    barcode = cell_ids,
    clusterlabel = clusterlabel,
    clusterlabel_refine = clusterlabel_refine
    )
    print("label_df")
    print(dim(label_df))
    output_dir <- "/home/cavin/jiangtao/python/wave/DLPFC/_SpatialPCA/cluterlabel_and_emb"
    output_file <- paste0(output_dir, "/", sample_names[i], "_spatial_cluster.csv")
    write.csv(label_df, output_file, row.names = FALSE)

    
    pcs_df <- data.frame(barcode = cell_ids, pcs)
    print("pcs_df")
    print(dim(pcs_df))

    output_file <- paste0(output_dir, "/", sample_names[i], "_spatial_emb.csv")
    write.csv(pcs_df, output_file, row.names = FALSE)

    # umap_embed <- LIBD@SpatialPCs %*% prcomp(LIBD@SpatialPCs)$rotation[, 1:3]  
    # print(dim(umap_embed))
    # umap_df <- data.frame(
    # cell_id = cell_ids,
    # UMAP1 = umap_embed[,1],
    # UMAP2 = umap_embed[,2],
    # UMAP3 = umap_embed[,3]
    # )
    # output_file <- paste0(output_dir, "/", sample_names[i], "_spatial_emb.csv")
    # write.csv(umap_df, output_file, row.names = FALSE)
}