from typing import Dict, List

import numpy as np
from hdbscan import HDBSCAN
from spacy.tokens.token import Token


def cluster_topics(nouns: List[Token], clustering_params: Dict[str, str]) -> HDBSCAN:
    vectors = convert_tokens_to_vector_df(nouns)
    clusterer = HDBSCAN(**clustering_params)
    clusterer.fit(vectors)
    return clusterer


def convert_tokens_to_vector_df(nouns: List[Token]) -> List[np.ndarray]:
    return [noun.vector for noun in nouns]
