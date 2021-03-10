import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import spacy
from sklearn.preprocessing import normalize
from spacy.tokens.span import Span
from spacy.tokens.token import Token

nlp = spacy.load("en_core_web_lg", disable=["ner"])
nlp.max_length = 2_000_000


def get_strings_and_vectors_from_text(
    text: str, downsample_factor: int = 1, max_duplicates: int = -1
) -> Tuple[List[str], List[np.ndarray]]:
    spans = get_noun_spans(text, downsample_factor=downsample_factor)
    text, vectors = get_strings_and_vectors_from_noun_spans(spans)
    vectors = normalize_vectors(vectors)
    if max_duplicates == -1:
        return text, vectors
    df = pd.DataFrame({"text": text, "vectors": vectors})
    df = df.groupby("text").head(max_duplicates)
    return df["text"].to_numpy().tolist(), df["vectors"].to_numpy().tolist()


def get_strings_and_vectors_from_noun_spans(
    noun_spans: List[Span],
) -> Tuple[List[str], List[np.ndarray]]:
    current_directory = os.path.dirname(os.path.abspath(__file__))
    glove_path = os.path.join(current_directory, "glove.6B.300d.txt")
    glove_embeddings = read_glove_embeddings(glove_path)
    text_and_vector = []
    for span in noun_spans:
        vector_list = []
        for token in span:
            text = token.lemma_.lower()
            vector = glove_embeddings.get(text, None)
            vector_list.append(vector)
        if all(vector is not None for vector in vector_list):
            embedding = np.vstack(vector_list).mean(axis=0)  # type: ignore
            text_and_vector.append((span.text, embedding))
    text, vectors = zip(*text_and_vector)
    return list(text), list(vectors)


def get_noun_spans(text: str, downsample_factor: int = 1) -> List[Span]:
    doc = nlp(text.lower())
    eligible_chunks = [
        phrase
        for phrase in doc.noun_chunks
        if all(func(token) for func in filter_funcs for token in phrase)
    ]
    return eligible_chunks[::downsample_factor]


def read_glove_embeddings(glove_path: str) -> Dict[str, np.ndarray]:
    embeddings_map = {}
    with open(glove_path, "r") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_map[word] = vector
    return embeddings_map


def normalize_vectors(vectors: List[np.ndarray]) -> List[np.ndarray]:
    vectors = normalize(vectors, norm="l2")
    return [vectors[i] for i in range(vectors.shape[0])]


def is_big_word(token: Token) -> bool:
    return len(token.text) > 2


def is_alpha(token: Token) -> bool:
    return token.is_alpha or token.text == "-"


def not_stop(token: Token) -> bool:
    return not token.is_stop


def is_probably_a_word(token: Token) -> bool:
    return not token.is_oov


def has_vector(token: Token) -> bool:
    return int(token.vector.sum()) != 0


def is_not_proper_noun(token: Token) -> bool:
    return token.pos_ != "PROPN"


filter_funcs = [
    is_big_word,
    is_alpha,
    not_stop,
    is_probably_a_word,
    has_vector,
    is_not_proper_noun,
]


if __name__ == "__main__":
    from data import read_text_from_txt

    text = read_text_from_txt("statistical_learning.txt")[0:1000]
    print(get_strings_and_vectors_from_text(text))
