# %%
from joblib import Memory

from clustering import cluster_topics
from data import read_text_from_txt
from nlp import get_noun_tokens
from plotting import get_plot_df, plot3d

text = read_text_from_txt("statistical_learning.txt")
nouns = get_noun_tokens(text, downsample_factor=10)
# %%
clustering_params = dict(
    min_cluster_size=40,
    min_samples=30,
    cluster_selection_epsilon=0,
    memory=Memory(".cache"),
)
clusterer = cluster_topics(nouns, clustering_params)

# %%
df = get_plot_df(clusterer, nouns, dims=3, add_noise=True, tsne=True, tsne_kwargs={})
fig = plot3d(df)
fig.show()
fig.write_html("./fig.html")
# %%
def add_rotation(fig):
    import numpy as np
    import plotly.graph_objects as go

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


fig = add_rotation(fig)

fig.write_html("./fig.html")

# %%
