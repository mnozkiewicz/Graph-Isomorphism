import numpy as np
import networkit as nk
import networkx as nx
from networkit.linkprediction import AdjustedRandIndex, JaccardIndex, CommonNeighborsIndex
from networkit.sparsification import TriangleEdgeScore, SCANStructuralSimilarityScore, LocalDegreeScore
from networkit.centrality import Betweenness


def jaccard_index(graph: nk.Graph, normalize: bool = True) -> np.ndarray:
    if normalize:
        jaccard_index = JaccardIndex(graph)
    else:
        jaccard_index = CommonNeighborsIndex(graph)
    scores = [jaccard_index.run(*edge) for edge in graph.iterEdges()]
    return np.array(scores)


def edge_betweenness(graph: nk.Graph, normalize: bool = True) -> np.ndarray:
    betweeness = Betweenness(graph, normalized=normalize, computeEdgeCentrality=True)
    betweeness.run()
    scores = betweeness.edgeScores()
    return np.array(scores)


def local_degree_score(graph: nk.Graph) -> np.ndarray:
    local_degree_score = LocalDegreeScore(graph)
    local_degree_score.run()
    scores = local_degree_score.scores()
    return np.array(scores)


def calculate_adjusted_rand_index(graph: nk.Graph) -> np.ndarray:
    index = AdjustedRandIndex(graph)
    scores = [index.run(u, v) for u, v in graph.iterEdges()]
    return np.array(scores)


def calculate_scan_structural_similarity_score(graph: nk.Graph) -> np.ndarray:
    triangles = TriangleEdgeScore(graph)
    triangles.run()
    triangles = triangles.scores()

    score = SCANStructuralSimilarityScore(graph, triangles)
    score.run()
    scores = score.scores()
    return np.array(scores)