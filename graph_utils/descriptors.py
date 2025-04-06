import numpy as np
import networkit as nk
import networkx as nx


def jaccard_index(graph: nk.Graph) -> np.ndarray:
    jaccard_index = nk.linkprediction.JaccardIndex(graph)
    scores = [jaccard_index.run(*edge) for edge in graph.iterEdges()]
    return np.array(scores)


def edge_betweenness(graph: nk.Graph) -> np.ndarray:
    betweeness = nk.centrality.Betweenness(graph, normalized=True, computeEdgeCentrality=True)
    betweeness.run()
    scores = np.array(betweeness.edgeScores())
    return scores


def local_degree_score(graph: nk.Graph) -> np.ndarray:
    local_degree_score = nk.sparsification.LocalDegreeScore(graph)
    local_degree_score.run()
    scores = local_degree_score.scores()
    return scores