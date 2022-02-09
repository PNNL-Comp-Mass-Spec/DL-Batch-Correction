import torch

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy.stats import f as fisher_dist
from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


## Batch effect testing functions


def test_batch_effect(y, n_batches, batch_size):
    p_values = list()

    for yy in y:
        d = {'value' : yy, 'batch' : [format(b) for b in range(n_batches) for i in range(batch_size)]}
        dff = pd.DataFrame(data = d)
        model = ols('value ~ batch', data = dff).fit()
        aov_table = sm.stats.anova_lm(model, typ = 2)
        p_value = aov_table.iloc[0,3]
        p_values.append(p_value)

    return(p_values)


def test_batch_effect_fast(y, n_batches, batch_size):
    length = len(y)
    y_mean = torch.mean(y, 1).view(length, 1).repeat_interleave(n_batches * batch_size, 1)

    y_batch_mean = y.view(length, n_batches, batch_size)
    y_batch_mean = torch.mean(y_batch_mean, 2).repeat_interleave(batch_size, 1)

    exp_var = torch.sum(torch.square(y_batch_mean - y_mean), 1)
    unexp_var = torch.sum(torch.square(y - y_batch_mean), 1)

    N = batch_size * n_batches
    K = n_batches

    F_stat = (exp_var/unexp_var) * ((N-K) / (K-1))
    p_values = 1 - fisher_dist.cdf(F_stat, dfn = K-1, dfd = N-K)

    return(p_values)    




def make_report(data, p_values, n_batches, batch_size, prefix = "", suffix = ""):
    row_names = data.index
    col_names = data.columns
    data = pd.DataFrame(StandardScaler().fit_transform(data))
    data.index = row_names
    data.columns = col_names
    data = data.transpose()
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(data)

    colors = ["#F8766D", "#B79F00", "#00BA38", "#00BFC4", "#4A528E", "#F564E3", "#939496"]
    sample_colors = [colors[i//batch_size] for i in range(batch_size * n_batches)]

    plt.scatter(pca_components[:, 0], pca_components[:, 1], c = sample_colors)
    plot_title = prefix + " PCA plot by batch, epoch " + suffix
    plt.title(plot_title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    path = "./pca_plots/" + plot_title + ".png"
    plt.savefig(path)

    plt.clf()

    plt.hist(p_values)
    plot_title = prefix + " p value histogram of batch effect, epoch " + suffix
    plt.title(plot_title)
    plt.ylabel('Count')
    plt.xlabel('p value')
    path = "./p_value_histograms/" + plot_title + ".png"
    plt.savefig(path)
    plt.clf()

    return "Saved plots"


