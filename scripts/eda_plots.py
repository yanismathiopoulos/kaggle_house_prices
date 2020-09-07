from scipy import stats
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_fig_matrix(features_to_vis, ncols=4):
    nrows = math.ceil(len(features_to_vis) / ncols)
    charts_last_row = len(features_to_vis) % ncols

    if charts_last_row == 0:
        extra_none = 0
    else:
        extra_none = ncols - charts_last_row

    features_to_vis_matrix = np.array(
        features_to_vis + [None for x in range(extra_none)]
    ).reshape(nrows, ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))

    return features_to_vis_matrix, fig, axes, nrows, ncols


def scatter_grid(df, features_to_vis, y):
    features_to_vis_matrix, fig, axes, nrows, ncols = get_fig_matrix(
        features_to_vis)

    for i in range(nrows):
        for j in range(ncols):
            try:
                sns.scatterplot(x=features_to_vis_matrix[i, j],
                                y=y,
                                data=df,
                                ax=axes[i, j])
            except ValueError:
                pass


def barplot_grid(df, features_to_vis, y):
    features_to_vis_matrix, fig, axes, nrows, ncols = get_fig_matrix(
        features_to_vis)

    for i in range(nrows):
        for j in range(ncols):
            try:
                if features_to_vis_matrix[i, j] is not None:
                    sns.barplot(x=features_to_vis_matrix[i, j],
                                y=y,
                                data=df,
                                ax=axes[i, j])
            except ValueError:
                pass


def distplot_grid(df, features_to_vis):
    features_to_vis_matrix, fig, axes, nrows, ncols = get_fig_matrix(
        features_to_vis)

    for i in range(nrows):
        for j in range(ncols):
            try:
                sns.distplot(df[features_to_vis_matrix[i, j]],
                             fit=stats.norm,
                             ax=axes[i, j])
            except KeyError:
                pass


def normplot_grid(df, features_to_vis):
    features_to_vis_matrix, fig, axes, nrows, ncols = get_fig_matrix(
        features_to_vis)

    z = 0
    for i in range(nrows):
        for j in range(ncols):
            z += 1
            try:
                plt.subplot(nrows, ncols, z)
                stats.probplot(
                    df[features_to_vis_matrix[i, j]],
                    plot=plt)
            except KeyError:
                pass


def check_normality(y):
    sns.distplot(y, fit=stats.norm)
    fig = plt.figure()
    res = stats.probplot(y, plot=plt)

