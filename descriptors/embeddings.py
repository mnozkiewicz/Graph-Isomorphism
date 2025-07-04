from tkinter import NO
from typing import Callable, Optional
from functools import partial
import numpy as np
import networkit as nk
from .edge_descriptors import jaccard_index, edge_betweenness, local_degree_score, calculate_adjusted_rand_index, calculate_scan_structural_similarity_score
from .node_descriptors import local_degree_profile


def get_function(name: str, normalize: bool) -> Callable[[nk.Graph], np.ndarray | list[np.ndarray]]:
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
        case _:
            raise ValueError(f"Unknown function name: {name}")


def normalize_features(features):
    distinct_features = []
    for feature in features:
        if feature == "moltop":
            distinct_features.extend(["ari", "scan", "edge_betweenness"])
        elif feature == "ltp":
            distinct_features.extend(["jaccard_index", "edge_betweenness", "lds"])
        else:
            distinct_features.append(feature)
    return sorted(set(distinct_features))

def create_embedding_function(
        features: list[str],
        bins_per_feature: int,
        histogram_range: Optional[tuple[int, int]] = None,
        include_node_features: bool = True,
        normalize: bool = True
    ) -> Callable[[nk.Graph], np.ndarray]:

    distinct_features = normalize_features(features)
    feature_functions = list(map(lambda x: get_function(x, normalize=normalize), distinct_features))
    

    def combined_features(graph: nk.Graph) -> np.ndarray:
        graph.indexEdges()

        edge_features = map(lambda f: f(graph), feature_functions)
        edge_histograms = [np.histogram(edge_feature, bins=bins_per_feature, range=histogram_range)[0] for edge_feature in edge_features]
        embedding = np.concatenate(edge_histograms)

        if include_node_features:
            ldp = local_degree_profile(graph)
            node_histograms = [np.histogram(feature, bins=bins_per_feature, range=histogram_range)[0] for feature in ldp]
            node_embedding = np.concatenate(node_histograms)
            embedding = np.concatenate((embedding, node_embedding))

        return embedding

    return combined_features
