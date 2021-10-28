
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compute_pca(x, y):
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import zscore
    from scipy.linalg import eigh
    
    x = zscore(x)
    A = np.matmul(x.T , x)

    k = x.shape[1]
    values, vectors = eigh(A, eigvals=(k-2, k-1))
    vectors = vectors.T
    T = np.matmul(vectors, x.T)

    T = np.vstack((T, y)).T
    df = pd.DataFrame(data=T, columns=('PC1', 'PC2', 'label'))
    df['label'] = df['label'].astype('category')
    return df
        
def plot_several_pca(datasets, y):
    fig, axs = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 6))
    for i in range(len(datasets)):
        key = list(datasets)[i]
        x = datasets[key]
        df = compute_pca(x, y)
        ax = axs[i]
        sns.scatterplot(x='PC1', y='PC2', data=df, hue='label', legend=False, ax=ax)
        ax.set_title(key)
    return axs

def plot_several_anova(datasets, y):
    from sklearn import feature_selection
    fig, axs = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 6))

    for i in range(len(datasets)):
        key = list(datasets)[i]
        x = datasets[key]
        y = labels
        f, p  = feature_selection.f_classif(x, y)
        ax = axs[i]
        ax.hist(p,
                bins = min(20, 1+int(len(p)/20)),
                color="grey")
        ax.set_xlim([0,1])
        ax.set_title('{} (N = {})'.format(key, len(p)))
    return axs