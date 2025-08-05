from functools import partial
from typing import Callable, List, Optional, Tuple

import networkit as nk
import numpy as np

from .edge_descriptors import (
    calculate_adjusted_rand_index,
    calculate_scan_structural_similarity_score,
    edge_betweenness,
    jaccard_index,
    local_degree_score,
)
from .node_descriptors import degree_ldp, max_ldp, mean_ldp, min_ldp, std_ldp


def get_function(
    name: str, normalize: bool
) -> Callable[[nk.Graph], np.ndarray | list[np.ndarray]]:
    match name:
        case "jaccard_index":
            return partial(jaccard_index, normalize=normalize)
        case "edge_betweenness":
            return partial(edge_betweenness, normalize=normalize)
        case "lds":
            return local_degree_score
        case "ari":
            return calculate_adjusted_rand_index
        case "scan":
            return calculate_scan_structural_similarity_score
        case "ldp_degree":
            return degree_ldp
        case "ldp_min":
            return min_ldp
        case "ldp_max":
            return max_ldp
        case "ldp_mean":
            return mean_ldp
        case "ldp_std":
            return std_ldp
        case _:
            raise ValueError(f"Unknown function name: {name}")


def normalize_features(features):
    distinct_features = []
    for feature in features:
        if feature == "moltop":
            distinct_features.extend(["ari", "scan", "edge_betweenness"])
        elif feature == "ltp":
            distinct_features.extend(["jaccard_index", "edge_betweenness", "lds"])
        elif feature == "ldp":
            distinct_features.extend(
                ["ldp_degree", "ldp_min", "ldp_max", "ldp_mean", "ldp_std"]
            )
        else:
            distinct_features.append(feature)
    return sorted(set(distinct_features))


def create_embedding_function(
    features: list[str],
    bins_per_feature: int,
    histogram_ranges: Optional[List[Tuple[int, int]]] = None,
    normalize: bool = True,
    embeddings: bool = True,  # if set to False, function returns raw values of function
) -> Callable[[nk.Graph], np.ndarray | List[np.ndarray]]:

    distinct_features = normalize_features(features)
    print(features, distinct_features)

    feature_functions = list(
        map(lambda x: get_function(x, normalize=normalize), distinct_features)
    )

    def combined_features(graph: nk.Graph) -> np.ndarray | List[np.ndarray]:
        graph.indexEdges()

        edge_features = list(map(lambda f: f(graph), feature_functions))

        edge_features_count = len(edge_features)
        if embeddings:
            edge_histograms = [
                np.histogram(edge_feature, bins=bins_per_feature, range=hrange)[0]
                for edge_feature, hrange in zip(
                    edge_features, histogram_ranges[:edge_features_count]
                )
            ]
            embedding = np.concatenate(edge_histograms)

        return embedding if embeddings else edge_features

    return combined_features
