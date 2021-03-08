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


def add_pca_columns(df: pd.DataFrame, exclude_noise=True, tsne=False) -> pd.DataFrame:
    if exclude_noise:
        vectors_for_fitting = df[df["label"] != -1]["vector"].to_list()
    else:
        vectors_for_fitting = df["vector"].to_list()
    fit_vectors = np.vstack(vectors_for_fitting)
    transform_vectors = np.vstack(df["vector"].to_list())
    if not tsne:
        pca = PCA(n_components=2)
        pca.fit(fit_vectors)
        new_dims = pca.transform(transform_vectors)
    else:
        tsne = TSNE(n_components=2)
        tsne.fit(fit_vectors)
        new_dims = tsne.transform(transform_vectors)
    df["pc1"], df["pc2"] = new_dims[:, 0], new_dims[:, 1]
    return df


def get_plot_df(
    clusterer: HDBSCAN, nouns: List[Token], add_noise: bool = True
) -> pd.DataFrame:
    noun = [noun.text for noun in nouns]
    vector = [noun.vector for noun in nouns]
    label = clusterer.labels_
    df = pd.DataFrame({"noun": noun, "vector": vector, "label": label})
    df = add_pca_columns(df)
    max_duplicates = 2
    df = df.groupby("noun").head(max_duplicates)
    if add_noise:
        return add_noise_to_plot_df(df)
    else:
        return df


def add_noise_to_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    mu, sigma = 0, 0.1
    noise = np.random.normal(mu, sigma, df[["pc1", "pc2"]].shape)
    df[["pc1", "pc2"]] = df[["pc1", "pc2"]] + noise
    return df


def plot(
    df: pd.DataFrame,
    plot_noise: bool = True,
    pca_noise: bool = False,
    vline: bool = True,
    hline: bool = True,
    cluster_circles: List[int] = [],
    word_vectors: List[int] = [],
    centroid_vectors: List[int] = [],
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
            x=sub_df["pc1"],
            y=sub_df["pc2"],
            mode="markers",
            text=sub_df["hover_text"],
            name=name,
            meta=name,
            hoverinfo="text",
            showlegend=True,
            marker=dict(color=color, opacity=opacity),
        )
        fig.add_trace(scatter_trace)

    # if vline:
    #     fig = self._add_vline(fig)
    # if hline:
    #     fig = self._add_hline(fig)

    # fig = self._add_cluster_circles(fig, cluster_labels=cluster_circles)
    # fig = self._draw_word_vectors(fig, cluster_labels=word_vectors)
    # fig = self._draw_centroid_vectors(fig, cluster_labels=centroid_vectors)
    # fig.layout.legend.title = "Topic Clusters"

    fig.layout.template = "plotly_white"
    fig.show()
    return fig


class ClusterPlotter:
    def get_plotting_df(self, pca_noise):
        df = pd.DataFrame()
        df["Original Vectors"] = self.vectors
        df["Word"] = self.words
        df["Cluster"] = self.labels

        pca = PCA(n_components=2)
        if pca_noise:
            fit_vectors = list(df["Original Vectors"].values)
        else:
            fit_vectors = list(df[df["Cluster"] != -1]["Original Vectors"].values)

        pca.fit(fit_vectors)
        transform_vectors = list(df["Original Vectors"].values)
        self.plot_vectors = pca.transform(transform_vectors)
        pc1, pc2 = zip(*self.plot_vectors)
        df["Principal Component 1"] = pc1
        df["Principal Component 2"] = pc2

        most_common_words = [self[label].most_common for label in self.labels]
        df["Topic"] = most_common_words
        df["Cluster"] = df["Cluster"].astype(str)
        return df

    def plot(
        self,
        plot_noise=True,
        pca_noise=False,
        vline=True,
        hline=True,
        cluster_circles=[],
        word_vectors=[],
        centroid_vectors=[],
    ):
        df = self.get_plotting_df(pca_noise)

        if not plot_noise:
            df = df[df["Cluster"] != "-1"]

        fig = go.Figure()

        df["hover_text"] = df[["Word", "Cluster", "Topic"]].apply(
            lambda row: f"{row['Word']}<br>Topic {row['Cluster']}: {row['Topic']}",
            axis=1,
        )

        colors = px.colors.qualitative.Light24
        color_index = 0
        for cluster in self:
            sub_df = df[df["Cluster"] == str(cluster.label)]

            if cluster.label == -1:
                color = "grey"
                opacity = 0.1
                name = "Noise"
            else:
                try:
                    color = colors[color_index]
                except IndexError:
                    color_index = 0
                    color = colors[color_index]

                opacity = None
                name = f"Topic {cluster.label}: {cluster.most_common}"

            scatter_trace = go.Scatter(
                x=sub_df["Principal Component 1"],
                y=sub_df["Principal Component 2"],
                mode="markers",
                text=sub_df["hover_text"],
                name=name,
                meta=cluster.label,
                hoverinfo="text",
                showlegend=True,
                marker=dict(color=color, opacity=opacity),
            )
            fig.add_trace(scatter_trace)

            color_index += 1

        if vline:
            fig = self._add_vline(fig)
        if hline:
            fig = self._add_hline(fig)

        fig = self._add_cluster_circles(fig, cluster_labels=cluster_circles)
        fig = self._draw_word_vectors(fig, cluster_labels=word_vectors)
        fig = self._draw_centroid_vectors(fig, cluster_labels=centroid_vectors)
        fig.layout.legend.title = "Topic Clusters"

        fig.layout.template = "plotly_white"
        return type(fig)

    def _add_vline(self, fig):
        fig.add_vline(
            x=0, visible=True, opacity=0.8, line=dict(dash="dash", color="grey")
        )
        return fig

    def _add_hline(self, fig):
        fig.add_hline(
            y=0, visible=True, opacity=0.8, line=dict(dash="dash", color="grey")
        )
        return fig

    def _cluster_bounds(self, cluster_label):
        labels = self.labels
        circle_vectors = [
            self.plot_vectors[i]
            for i, label in enumerate(labels)
            if label == cluster_label
        ]

        x_vals, y_vals = zip(*circle_vectors)

        xmin = min(x_vals)
        xmax = max(x_vals)
        ymin = min(y_vals)
        ymax = max(y_vals)
        return xmin, xmax, ymin, ymax

    def _add_cluster_circles(self, fig, cluster_labels=[]):
        for cluster_label in cluster_labels:
            xmin, xmax, ymin, ymax = self._cluster_bounds(cluster_label)
            color = [
                data["marker"]["color"]
                for data in fig.to_dict()["data"]
                if data["meta"] == cluster_label
            ][0]
            cluster_name = self[cluster_label].most_common
            fig.add_shape(
                type="circle",
                xref="x",
                yref="y",
                x0=xmin - 0.3,
                y0=ymin - 0.3,
                x1=xmax + 0.3,
                y1=ymax + 0.3,
                opacity=0.2,
                fillcolor=color,
                line_color=color,
            )
            fig.add_annotation(x=xmin, y=ymax, text=cluster_name)
        return fig

    def _draw_vector(self, fig, x, y):
        fig.add_annotation(
            x=x,
            y=y,
            ax=0,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text="",
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="black",
        )
        return fig

    def _get_vectors_for_label(self, cluster_label):
        return [
            vector
            for label, vector in zip(self.labels, self.plot_vectors)
            if label == cluster_label
        ]

    def _draw_word_vectors(self, fig, cluster_labels=[]):
        if not cluster_labels:
            return fig
        for cluster_label in cluster_labels:
            current_vectors = self._get_vectors_for_label(cluster_label)
            for current_vector in current_vectors:
                x, y = current_vector
                self._draw_vector(fig, x, y)
        return fig

    def _draw_centroid_vectors(self, fig, cluster_labels=[]):
        if not cluster_labels:
            return fig
        for cluster_label in cluster_labels:
            xmin, xmax, ymin, ymax = self._cluster_bounds(cluster_label)
            x, y = np.mean([xmin, xmax]), np.mean([ymin, ymax])
            self._draw_vector(fig, x, y)
        return fig
