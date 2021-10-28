
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compute_pca(x):
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import zscore
    from scipy.linalg import eigh
    
    x = zscore(x)
    k = x.shape[1]
    cov_mat = np.cov(x , rowvar = False)
    eigenvalues , eigenvectors = eigh(cov_mat, eigvals=(k-2, k-1))
    #eigenvectors = eigenvectors.T
    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:,idx]
    eigenvector_subset = sorted_eigenvectors[:,0:2]
    z = np.dot(eigenvector_subset.T, x.T)
    return z.T
        
def plot_several_pca(datasets, y):
    fig, axs = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 6))
    y = np.expand_dims(y, 1)
    for i in range(len(datasets)):
        key = list(datasets)[i]
        x = datasets[key]
        z = compute_pca(x)
        df = pd.DataFrame(np.concatenate([z, y], axis=1), columns = ['PC1','PC2','batch'])
        df['batch'] = df['batch'].astype('category')
        sns.scatterplot(x='PC1', y='PC2', data=df, hue='batch', legend=False, ax=axs[i])
        axs[i].set_title(key)
    return axs

def plot_several_anova(datasets, y):
    from sklearn import feature_selection
    fig, axs = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 6))

    for i in range(len(datasets)):
        key = list(datasets)[i]
        x = datasets[key]
        f, p  = feature_selection.f_classif(x, y)
        ax = axs[i]
        ax.hist(p,
                bins = min(20, 1+int(len(p)/20)),
                color="grey")
        ax.set_xlim([0,1])
        ax.set_title('{} (N = {})'.format(key, len(p)))
    return axs