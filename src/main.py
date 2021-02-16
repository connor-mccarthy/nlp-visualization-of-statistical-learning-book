from clustering import cluster_topics
from data import get_text_from_url
from plotting import get_plot_df
from vectorize import get_noun_tokens


def main():  # type: ignore
    book_url = "http://www.africau.edu/images/default/sample.pdf"
    pages = get_text_from_url(book_url)
    nouns = get_noun_tokens(pages)
    clusterer = cluster_topics(nouns)
    return get_plot_df(clusterer, nouns)


if __name__ == "__main__":
    print(main())  # type: ignore
