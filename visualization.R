


library(optparse)
library(MSnSet.utils)
library(ggplot2)
library(ggpubr)
library(dplyr)

option_list <- list(
  make_option(c("-p", "--path_to_data"), type="character", default=NULL, 
              help="Path to model outputs", metavar="character")
  )

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

output_dir <- opt$path_to_data

# for testing
# path_to_data <- "./8-31-pipeline-test/original_test_data.csv"
# path_to_labels <- "./8-31-pipeline-test/test_labels.csv"

load_csv <- function(path) {
  x <- read.csv(path, header=T, check.names=F)
  rownames(x) <- x[,1]
  x <- x[,2:ncol(x),drop=F]
}

load_msnset <- function(path_to_data, path_to_labels) {
  x <- load_csv(path_to_data)
  y <- load_csv(path_to_labels)
  y$batch <- as.factor(y$batch)
  m <- MSnSet(as.matrix(x), pData=y)
}

path_to_test_labels <- file.path(output_dir, "test_labels.csv")
test_data <-  c("original_test_data.csv",
                "encoded_test_data.csv",
                "reconstructed_test_data.csv",
                "corrected_test_data.csv") %>%
  file.path(output_dir, .) %>%
  lapply(load_msnset, path_to_test_labels)

path_to_train_labels <- file.path(output_dir, "train_labels.csv")
train_data <-  c("original_train_data.csv",
                "encoded_train_data.csv",
                "reconstructed_train_data.csv",
                "corrected_train_data.csv") %>%
  file.path(output_dir, .) %>%
  lapply(load_msnset, path_to_train_labels)


plot_pca <- function(msnset_list, output_file) {
  plts <- suppressWarnings(lapply(msnset_list, plot_pca_v3, "batch"))
  
  library(ggpubr)
  p <- ggarrange(plotlist=plts,labels = c("Original", "Encoded", "Reconstructed", "Corrected"),
            common.legend=T)
  pdf(file=file.path(output_dir, output_file), width=12, height=12)
  print(p)
  dev.off()
}

plot_pca(test_data, "PCA_test_data.pdf")
plot_pca(train_data,"PCA_train_data.pdf")

plot_pvalue_histogram <- function(msnset_list, output_file) {
  plts <- lapply(msnset_list, function(m) {
    res <- limma_gen(m, "~batch", "batch")
    ggplot(res, aes(x=P.Value)) + geom_histogram(breaks=seq(0,1,0.05)) + xlim(0,1)
  })
  
  p <- ggarrange(plotlist=plts,labels = c("Original", "Encoded", "Reconstructed", "Corrected"),
                 common.legend=T)
  pdf(file=file.path(output_dir, output_file), width=6, height=6)
  print(p)
  dev.off()
}

plot_pvalue_histogram(test_data, "pvalue_histogram_test_data.pdf")
plot_pvalue_histogram(train_data,"pvalue_histogram_train_data.pdf")



