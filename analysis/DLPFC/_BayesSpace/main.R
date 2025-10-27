

library(SingleCellExperiment)
library(ggplot2)
library(BayesSpace)
library(Seurat)
library(scuttle)

data_path <- "E:\\spatial_data\\data\\DLPFC"
save_path <- "C:\\Users\\tao\\Desktop\\python\\wave\\DLPFC\\_BayesSpace\\images"
data_path <- "/home/cavin/jiangtao/spatial_data/visium_spaceranger_out"
save_path <- "/home/cavin/jiangtao/python/wave/DLPFC/_BayesSpace/images"


# sample <- c('Section1_Posterior', 'Section1_Anterior', 'Section_Coronal', "Human_breast_FFPE")

sample <- c('151507', '151508','151509','151510', '151669', '151670', '151671', '151672', '151673','151674','151675','151676')
for (sample.name in sample ){
  print(sample.name)
dir.input <- file.path(data_path, sample.name)
# dir.output <- file.path(save_path, sample.name)

# if(!dir.exists(file.path(dir.output))){
#   dir.create(file.path(dir.output), recursive = TRUE)
# }

if(sample.name %in% c('151669', '151670', '151671', '151672')) {
  n_clusters <- 5} else {
  n_clusters <- 7}

### load data
dlpfc <- readVisium(dir.input) 
print("load data finished")
dlpfc <- logNormCounts(dlpfc)

set.seed(88)
dec <- scran::modelGeneVar(dlpfc)
top <- scran::getTopHVGs(dec, n = 3000)

set.seed(66)
dlpfc <- scater::runPCA(dlpfc, subset_row=top)

## Add BayesSpace metadata
dlpfc <- spatialPreprocess(dlpfc, platform="Visium", skip.PCA=TRUE)

q <- n_clusters  # Number of clusters
d <- 15  # Number of PCs

## Run BayesSpace clustering
set.seed(104)
dlpfc <- spatialCluster(dlpfc, q=q, d=d, platform='Visium',
                        nrep=50000, gamma=3, save.chain=TRUE)

labels <- dlpfc$spatial.cluster

# ## View results
# clusterPlot(dlpfc, label=labels, palette=NULL, size=0.05) +
#   scale_fill_viridis_d(option = "A", labels = 1:7) +
#   labs(title="BayesSpace")

# ggsave(file.path(dir.output, 'clusterPlot.png'), width=5, height=5)

##### save label

dir.out = paste0(save_path,"/", sample.name, "_label.csv")
label_df <- data.frame(barcode = colnames(dlpfc), labels)
write.csv(label_df, dir.out, row.names = FALSE)
# write.table(colData(dlpfc), file=file.path(dir.output, 'bayesSpace.csv'), sep='\t', quote=FALSE)

}
