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


def jaccard_index(graph: nk.Graph, normalize: bool = True) -> np.ndarray:
    if normalize:
        jaccard_index = JaccardIndex(graph)
    else:
        jaccard_index = CommonNeighborsIndex(graph)
    scores = [jaccard_index.run(*edge) for edge in graph.iterEdges()]
    return np.array(scores, np.float16)


def edge_betweenness(graph: nk.Graph, normalize: bool = True) -> np.ndarray:
    betweeness = Betweenness(graph, normalized=normalize, computeEdgeCentrality=True)
    betweeness.run()
    scores = betweeness.edgeScores()
    return np.array(scores, np.float16)


def local_degree_score(graph: nk.Graph) -> np.ndarray:
    local_degree_score = LocalDegreeScore(graph)
    local_degree_score.run()
    scores = local_degree_score.scores()
    return np.array(scores, np.float16)


def calculate_adjusted_rand_index(graph: nk.Graph) -> np.ndarray:
    index = AdjustedRandIndex(graph)
    scores = [index.run(u, v) for u, v in graph.iterEdges()]
    return np.array(scores, np.float16)


def calculate_scan_structural_similarity_score(graph: nk.Graph) -> np.ndarray:
    triangles = TriangleEdgeScore(graph)
    triangles.run()
    triangles = triangles.scores()

    score = SCANStructuralSimilarityScore(graph, triangles)
    score.run()
    scores = score.scores()
    return np.array(scores, np.float16)
