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
        pca = PCA(n_components=3)
        pca.fit(fit_vectors)
        new_dims = pca.transform(transform_vectors)
    else:
        pca = PCA(n_components=50)
        pca_dims = pca.fit(fit_vectors)
        pca_dims = pca.transform(transform_vectors)
        tsne = TSNE(n_components=3, **tsne_kwargs)
        new_dims = tsne.fit_transform(pca_dims)
    if not transform_noise:
        df = df[df["label"] != -1]

    df["x"], df["y"], df["z"] = new_dims[:, 0], new_dims[:, 1], new_dims[:, 2]
    return df


def get_plot_df(
    text: List[str],
    vectors: List[np.array],
    labels: List[int],
    add_noise: bool = True,
    fit_noise=False,
    transform_noise=True,
    tsne=False,
    tsne_kwargs={},
) -> pd.DataFrame:
    df = pd.DataFrame({"text": text, "vector": vectors, "label": labels})
    df = add_pca_columns(
        df,
        fit_noise=fit_noise,
        transform_noise=transform_noise,
        tsne=tsne,
        tsne_kwargs=tsne_kwargs,
    )
    max_duplicates = 50
    df = df.groupby("text").head(max_duplicates)
    if add_noise:
        return add_noise_to_plot_df(df)
    else:
        return df


def add_noise_to_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    mu, sigma = 0, 0.01
    noise = np.random.normal(mu, sigma, df[["x", "y", "z"]].shape)
    df[["x", "y", "z"]] = df[["x", "y", "z"]] + noise
    return df


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
        else:
            try:
                color = colors[color_index]
                color_index += 1
            except IndexError:
                color_index = 0
                color = colors[color_index]

            opacity = None

            # frequency_list = sorted(
            #     dict(Counter(sub_df["text"].values)).items(), key=lambda x: x[1]
            # )
            # most_common = frequency_list[0][0]

        scatter_trace = go.Scatter3d(
            x=sub_df["x"],
            y=sub_df["y"],
            z=sub_df["z"],
            mode="markers",
            text=sub_df["text"],
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


def add_rotation(fig):
    x_eye = 0
    y_eye = 1
    z_eye = 0

    fig.update_layout(
        title="Animation Test",
        width=600,
        height=600,
        scene_camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1,
                x=0.8,
                xanchor="left",
                yanchor="bottom",
                pad=dict(t=45, r=10),
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=5, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    ),
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
            )
        ],
    )

    def rotate_z(x, y, z, theta):
        w = x + 1j * y
        return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

    frames = []
    for t in np.arange(0, 6.26, 0.005):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=ze))))

    fig.frames = frames + frames
    return fig


def add_zoom(fig):
    import numpy as np
    import plotly.graph_objects as go

    x_eye = 0
    y_eye = 2
    z_eye = 0

    fig.update_layout(
        title="Animation Test",
        width=600,
        height=600,
        scene_camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1,
                x=0.8,
                xanchor="left",
                yanchor="bottom",
                pad=dict(t=45, r=10),
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=1, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    )
                ],
            )
        ],
    )
    frames = []
    for i in np.arange(y_eye, 1, -0.05):
        xe, ye, ze = x_eye, i, z_eye
        frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=ze))))

    y_eye = i

    def rotate_z(x, y, z, theta):
        w = x + 1j * y
        return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

    for t in np.arange(0, 6.26, 0.001):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=ze))))
    fig.frames = frames
    return fig
