import numpy as np
import networkit as nk
import networkx as nx
from functools import wraps

def _calculate_degress(graph):
    adj = nx.to_scipy_sparse_array(nk.nxadapter.nk2nx(graph), format='coo', dtype=float)
    return adj.sum(axis=1), adj

def _calculate_degree_matrix(graph, normalize):
    degrees, adj = _calculate_degress(graph)
    degree_matrix = adj * degrees
    if normalize:
        degree_matrix = degree_matrix / degrees.shape[0]

    return degree_matrix, degrees

def change_to_numpy(function, *args, **kwargs):
    @wraps(function)
    def wrapper(*args, **kwargs):
        return np.array(function(*args, **kwargs), dtype=np.float16)
    return wrapper

def local_degree_profile(graph: nk.Graph, normalize: bool = True) -> list[np.ndarray]:
    adj = nx.to_scipy_sparse_array(nk.nxadapter.nk2nx(graph), format='coo', dtype=float)
    degrees = adj.sum(axis=1)

    dn = adj * degrees
    if normalize:
        dn = dn / degrees.shape[0]

    min_dn = dn.min(axis=1, explicit=True).toarray()
    max_dn = dn.max(axis=1, explicit=True).toarray()
    mean_dn = dn.sum(axis=1) / degrees

    std_dn = (dn**2).sum(axis=1) / degrees - mean_dn**2

    ldp = [
        np.array(degrees / degrees.shape[0], np.float16), 
        np.array(min_dn, np.float16), 
        np.array(max_dn, np.float16),
        np.array(mean_dn, np.float16),
        np.array(std_dn, np.float16)
    ]
    return ldp

def degree_ldp(graph: nk.Graph, normalize: bool = True) -> np.ndarray:
    degrees, _ = _calculate_degress(graph)
    return degrees / degrees.shape[0]

@change_to_numpy
def min_ldp(graph: nk.Graph, normalize: bool = True) -> np.ndarray:
    degree_matrix, _ = _calculate_degree_matrix(graph, normalize)
    return degree_matrix.min(axis=1, explicit=True).toarray()


@change_to_numpy
def max_ldp(graph: nk.Graph, normalize: bool = True) -> np.ndarray:
    degree_matrix, _ = _calculate_degree_matrix(graph, normalize)
    return degree_matrix.max(axis=1, explicit=True).toarray()

@change_to_numpy
def mean_ldp(graph: nk.Graph, normalize: bool = True) -> np.ndarray:
    degree_matrix, degrees = _calculate_degree_matrix(graph, normalize)
    return  degree_matrix.sum(axis=1) / degrees

def std_ldp(graph: nk.Graph, normalize: bool = True) -> np.ndarray:
    degree_matrix, degrees = _calculate_degree_matrix(graph, normalize)
    mean_degree = degree_matrix.sum(axis=1) / degrees
    return (degree_matrix**2).sum(axis=1) / degrees - mean_degree**2