# %%
from joblib import Memory

from clustering import cluster_topics
from data import read_text_from_txt
from nlp import get_noun_tokens
from plotting import get_plot_df, plot

text = read_text_from_txt("statistical_learning.txt")[:1_000_000]
nouns = get_noun_tokens(text)

# %%
clustering_params = {
    "min_cluster_size": 100,
    "min_samples": 100,
    "cluster_selection_epsilon": 0,
    "memory": Memory(".cache"),
}
clusterer = cluster_topics(nouns, clustering_params)

# %%
df = get_plot_df(clusterer, nouns, add_noise=False)
fig = plot(df, plot_noise=False)
fig.show()
