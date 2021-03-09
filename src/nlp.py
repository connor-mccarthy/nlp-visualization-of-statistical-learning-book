import os
from typing import Dict, List, Tuple

import numpy as np
import spacy
from sklearn.preprocessing import normalize
from spacy.tokens.token import Token
from wordfreq import word_frequency

nlp = spacy.load("en_core_web_lg", disable=["ner", "parser"])
nlp.max_length = 2_000_000


def is_big_word(token: Token) -> bool:
    return len(token.text) > 2


def is_alpha(token: Token) -> bool:
    return token.is_alpha


def not_stop(token: Token) -> bool:
    return not token.is_stop


def is_probably_a_word(token: Token) -> bool:
    return not token.is_oov


def has_vector(token: Token) -> bool:
    return np.sum(token.vector) != 0


def is_common_word(token: Token) -> bool:
    return word_frequency(token.text, "en", "best" < 0.0005)


filter_funcs = [is_big_word, is_alpha, not_stop, is_probably_a_word, has_vector]


def get_noun_tokens(text: str, downsample_factor: int = 1) -> List[Token]:
    doc = nlp(text.lower())
    nouns = [token for token in doc if token.pos_ == "NOUN"]
    return [token for token in nouns if all(func(token) for func in filter_funcs)][
        ::downsample_factor
    ]


def read_glove_embeddings(glove_path: str) -> Dict[int, np.ndarray]:
    embeddings_map = {}
    with open(glove_path, "r") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_map[word] = vector
    return embeddings_map


def get_strings_and_vectors_from_tokens(
    tokens: List[spacy.tokens.token.Token],
) -> Tuple[np.ndarray, np.ndarray]:
    current_directory = os.path.dirname(os.path.abspath(__file__))
    glove_path = os.path.join(current_directory, "glove.6B.300d.txt")
    glove_embeddings = read_glove_embeddings(glove_path)
    text_and_vector = []
    for token in tokens:
        text = token.lemma_.lower()
        embedding = glove_embeddings.get(text, None)
        if embedding is not None:
            text_and_vector.append((text, embedding))
    text, vectors = zip(*text_and_vector)
    return list(text), list(vectors)


def normalize_vectors(vectors: List[np.ndarray]) -> List[np.ndarray]:
    vectors = normalize(vectors, norm="l2")
    return [vectors[i] for i in range(vectors.shape[0])]


def get_topics_and_vectors(
    text: str, downsample_factor: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    nouns = get_noun_tokens(text, downsample_factor=downsample_factor)
    text, vectors = get_strings_and_vectors_from_tokens(nouns)
    vectors = normalize_vectors(vectors)
    return text, vectors


if __name__ == "__main__":
    from data import read_text_from_txt

    text = read_text_from_txt("statistical_learning.txt")
    print(get_topics_and_vectors(text, downsample_factor=10))
