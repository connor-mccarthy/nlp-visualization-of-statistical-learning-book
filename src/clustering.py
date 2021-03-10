from typing import Any, Dict, List

import numpy as np
from hdbscan import HDBSCAN


def cluster_topics(
    vectors: List[np.ndarray], clustering_params: Dict[str, Any]
) -> HDBSCAN:
    clusterer = HDBSCAN(**clustering_params)
    clusterer.fit(vectors)
    return clusterer
