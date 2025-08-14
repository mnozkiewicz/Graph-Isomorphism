from functools import partial

import networkit as nk
import numpy as np
from networkit.centrality import Betweenness
from networkit.linkprediction import (
    AdjustedRandIndex,
    CommonNeighborsIndex,
    JaccardIndex,
)
from networkit.sparsification import (
    LocalDegreeScore,
    SCANStructuralSimilarityScore,
    TriangleEdgeScore,
)

edge_descriptors_dict = {}


def add_to_dict(name, can_be_normalized=False):
    def decorator(f):
        if can_be_normalized:
            edge_descriptors_dict[name] = partial(f, normalize=False)
            edge_descriptors_dict[name + "_normalized"] = partial(f, normalize=True)
        else:
            edge_descriptors_dict[name] = f

        return f

    return decorator


@add_to_dict("jaccard_index", can_be_normalized=True)
def jaccard_index(graph: nk.Graph, normalize: bool = True) -> np.ndarray:
    if normalize:
        jaccard_index = JaccardIndex(graph)
    else:
        jaccard_index = CommonNeighborsIndex(graph)
    scores = [jaccard_index.run(*edge) for edge in graph.iterEdges()]
    return np.array(scores, np.float16)


@add_to_dict("edge_betweenness", can_be_normalized=True)
def edge_betweenness(graph: nk.Graph, normalize: bool = True) -> np.ndarray:
    betweeness = Betweenness(graph, normalized=normalize, computeEdgeCentrality=True)
    betweeness.run()
    scores = betweeness.edgeScores()
    return np.array(scores, np.float16)


@add_to_dict("lds")
def local_degree_score(graph: nk.Graph) -> np.ndarray:
    local_degree_score = LocalDegreeScore(graph)
    local_degree_score.run()
    scores = local_degree_score.scores()
    return np.array(scores, np.float16)


@add_to_dict("ari")
def calculate_adjusted_rand_index(graph: nk.Graph) -> np.ndarray:
    index = AdjustedRandIndex(graph)
    scores = [index.run(u, v) for u, v in graph.iterEdges()]
    return np.array(scores, np.float16)


@add_to_dict("scan")
def calculate_scan_structural_similarity_score(graph: nk.Graph) -> np.ndarray:
    triangles = TriangleEdgeScore(graph)
    triangles.run()
    triangles = triangles.scores()

    score = SCANStructuralSimilarityScore(graph, triangles)
    score.run()
    scores = score.scores()
    return np.array(scores, np.float16)
