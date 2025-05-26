import numpy as np
import networkit as nk
import networkx as nx

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
