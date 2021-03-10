from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

Number = Union[float, int, np.imag, np.real]


def plot3d(
    df: pd.DataFrame,
    plot_noise: bool = True,
) -> go.Figure:

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
            opacity: Optional[float] = 0.1
            name = "Noise"
        else:
            try:
                color = colors[color_index]
                color_index += 1
            except IndexError:
                color_index = 0
                color = colors[color_index]
            opacity = None

        cluster_label = f"Cluster #{label + 1}" if label != -1 else "[Noise]"
        sub_df.loc[:, "hover_text"] = sub_df["text"].apply(
            lambda text: f"{cluster_label}<br>{text}<br>"
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

    axes = dict(title="", showgrid=False, zeroline=False, showticklabels=False)
    camera = dict(
        up=dict(x=0, y=0, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=1.2, y=0, z=0)
    )

    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(xaxis=axes, yaxis=axes, zaxis=axes),
        template="plotly_white",
        scene_camera=camera,
    )
    fig.update_layout(layout)

    return fig


def get_plot_df(
    text: List[str],
    vectors: List[np.ndarray],
    labels: List[int],
    add_vibration: bool = True,
    use_tsne: bool = False,
    tsne_kwargs: Dict[str, Any] = {},
) -> pd.DataFrame:
    df = pd.DataFrame({"text": text, "vector": vectors, "label": labels})
    df = add_dim_reduction_cols(
        df,
        use_tsne=use_tsne,
        tsne_kwargs=tsne_kwargs,
    )
    if add_vibration:
        return add_vibration_to_plot_df(df)
    else:
        return df


def add_dim_reduction_cols(
    df: pd.DataFrame,
    use_tsne: bool = False,
    tsne_kwargs: Dict[str, Any] = {},
) -> pd.DataFrame:
    fit_vectors = np.vstack(df[df["label"] != -1]["vector"].to_list())
    transform_vectors = np.vstack(df["vector"].to_list())

    if not use_tsne:
        pca = PCA(n_components=3)
        pca.fit(fit_vectors)
        new_dims = pca.transform(transform_vectors)
    else:
        pca = PCA(n_components=50)
        pca_dims = pca.fit(fit_vectors)
        pca_dims = pca.transform(transform_vectors)
        tsne = TSNE(n_components=3, **tsne_kwargs)
        new_dims = tsne.fit_transform(pca_dims)

    df["x"], df["y"], df["z"] = new_dims[:, 0], new_dims[:, 1], new_dims[:, 2]
    return df


def add_vibration_to_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    mu, sigma = 0, 0.01
    noise = np.random.normal(mu, sigma, df[["x", "y", "z"]].shape)
    df[["x", "y", "z"]] = df[["x", "y", "z"]] + noise
    return df
