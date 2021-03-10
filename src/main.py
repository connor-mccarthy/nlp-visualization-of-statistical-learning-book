# %%
import os

import plotly.graph_objects as go
from joblib import Memory

from clustering import cluster_topics
from data import read_text_from_txt
from nlp import get_strings_and_vectors_from_text
from plotting import get_plot_df, plot3d


def main() -> None:
    text = read_text_from_txt("statistical_learning.txt")
    topics, vectors = get_strings_and_vectors_from_text(text)
    hdbscan_params = dict(
        min_cluster_size=50,
        min_samples=40,
        cluster_selection_epsilon=0,
        memory=Memory(".cache"),
    )
    clusterer = cluster_topics(vectors, hdbscan_params)

    df = get_plot_df(
        text=topics,
        vectors=vectors,
        labels=clusterer.labels_,
        add_vibration=False,
        fit_noise=False,
        transform_noise=True,
        use_tsne=True,
        tsne_kwargs=dict(perplexity=30),
    )

    fig = plot3d(df, plot_noise=True)

    current_dir = os.path.dirname(os.path.abspath(__file__))

    fig.write_html(os.path.join(current_dir, "..", "figure.html"))
    fig.write_json(os.path.join(current_dir, "..", "figure.json"))


if __name__ == "__main__":
    main()
