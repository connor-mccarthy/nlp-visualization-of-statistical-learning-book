from typing import List

import numpy as np
from hdbscan import HDBSCAN
from spacy.tokens.token import Token


def cluster_topics(nouns: List[Token]) -> HDBSCAN:
    vectors = convert_tokens_to_vector_df(nouns)

    params = {}  # type: ignore
    clusterer = HDBSCAN(**params)
    clusterer.fit(vectors)
    return clusterer


def convert_tokens_to_vector_df(nouns: List[Token]) -> List[np.ndarray]:
    return [noun.vector for noun in nouns]
