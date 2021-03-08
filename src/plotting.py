from collections import Counter
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from spacy.tokens.token import Token


def add_pca_columns(
    df: pd.DataFrame,
    fit_noise=False,
    transform_noise=True,
    dims=2,
    tsne=False,
    tsne_kwargs={},
) -> pd.DataFrame:
    if fit_noise:
        fit_vectors = np.vstack(df["vector"].to_list())
    else:
        fit_vectors = np.vstack(df[df["label"] != -1]["vector"].to_list())

    if transform_noise:
        transform_vectors = np.vstack(df["vector"].to_list())
    else:
        transform_vectors = fit_vectors

    if not tsne:
        pca = PCA(n_components=dims)
        pca.fit(fit_vectors)
        new_dims = pca.transform(transform_vectors)
    else:
        pca = PCA(n_components=50)
        pca_dims = pca.fit(fit_vectors)
        pca_dims = pca.transform(transform_vectors)
        tsne = TSNE(n_components=dims, perplexity=30, **tsne_kwargs)
        new_dims = tsne.fit_transform(pca_dims)
    if not transform_noise:
        df = df[df["label"] != -1]

    if dims == 3:
        df["x"], df["y"], df["z"] = new_dims[:, 0], new_dims[:, 1], new_dims[:, 2]
    else:
        df["x"], df["y"] = new_dims[:, 0], new_dims[:, 1]

    return df


def get_plot_df(
    clusterer: HDBSCAN,
    nouns: List[Token],
    add_noise: bool = True,
    dims=2,
    tsne=False,
    tsne_kwargs={},
) -> pd.DataFrame:
    assert dims in [2, 3]
    noun = [noun.text for noun in nouns]
    vector = [noun.vector for noun in nouns]
    label = clusterer.labels_
    df = pd.DataFrame({"noun": noun, "vector": vector, "label": label})
    df = add_pca_columns(df, dims=dims, tsne=tsne, tsne_kwargs={})
    # max_duplicates = 2
    # df = df.groupby("noun").head(max_duplicates)
    if add_noise:
        return add_noise_to_plot_df(df, dims)
    else:
        return df


def add_noise_to_plot_df(df: pd.DataFrame, dims: int) -> pd.DataFrame:
    mu, sigma = 0, 0.05
    if dims == 2:
        noise = np.random.normal(mu, sigma, df[["x", "y"]].shape)
        df[["x", "y"]] = df[["x", "y"]] + noise
    if dims == 3:
        noise = np.random.normal(mu, sigma, df[["x", "y", "z"]].shape)
        df[["x", "y", "z"]] = df[["x", "y", "z"]] + noise

    return df


def plot2d(
    df: pd.DataFrame,
    plot_noise: bool = True,
):

    fig = go.Figure()

    colors = px.colors.qualitative.Light24
    color_index = 0
    unique_labels = list(sorted(set(df["label"])))
    for label in unique_labels:
        if not plot_noise and label == -1:
            continue
        sub_df = df[df["label"] == label]

        if label == -1:
            color = "grey"
            opacity = 0.1
            name = "Noise"
            hover_name = f"Topic {label}: {name}"
        else:
            try:
                color = colors[color_index]
                color_index += 1
            except IndexError:
                color_index = 0
                color = colors[color_index]

            opacity = None

            frequency_list = sorted(
                dict(Counter(sub_df["noun"].values)).items(), key=lambda x: x[1]
            )
            most_common = frequency_list[0][0]
            name = f"Topic {label}: {most_common}"
            hover_name = name

        sub_df["hover_text"] = sub_df["noun"].apply(
            lambda noun: f"{noun}<br>{hover_name}"
        )

        scatter_trace = go.Scatter(
            x=sub_df["x"],
            y=sub_df["y"],
            mode="markers",
            text=sub_df["hover_text"],
            name=name,
            meta=name,
            hoverinfo="text",
            showlegend=False,
            marker=dict(color=color, opacity=opacity),
        )
        fig.add_trace(scatter_trace)

    fig.layout.template = "plotly_white"
    return fig


def plot3d(
    df,
    plot_noise: bool = True,
):
    fig = go.Figure()
    colors = px.colors.qualitative.Light24
    color_index = 0

    unique_labels = list(sorted(set(df["label"])))
    for label in unique_labels:
        if not plot_noise and label == -1:
            continue
        sub_df = df[df["label"] == label]

        if label == -1:
            color = "grey"
            opacity = 0.1
            name = "Noise"
            hover_name = f"Topic {label}: {name}"
        else:
            try:
                color = colors[color_index]
                color_index += 1
            except IndexError:
                color_index = 0
                color = colors[color_index]

            opacity = None

            frequency_list = sorted(
                dict(Counter(sub_df["noun"].values)).items(), key=lambda x: x[1]
            )
            most_common = frequency_list[0][0]
            name = f"Topic {label}: {most_common}"
            hover_name = name

        sub_df["hover_text"] = sub_df["noun"].apply(
            lambda noun: f"{noun}<br>{hover_name}"
        )

        scatter_trace = go.Scatter3d(
            x=sub_df["x"],
            y=sub_df["y"],
            z=sub_df["z"],
            mode="markers",
            text=sub_df["hover_text"],
            name=name,
            meta=name,
            hoverinfo="text",
            showlegend=False,
            marker=dict(size=3, color=color, opacity=opacity),
        )
        fig.add_trace(scatter_trace)

    axes = dict(title="", showgrid=True, zeroline=False, showticklabels=False)

    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(xaxis=axes, yaxis=axes, zaxis=axes),
    )
    fig.update_layout(layout)
    fig.layout.template = "plotly_white"
    return fig
