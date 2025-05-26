import networkx as nx
import networkit as nk
from typing import Generator, List, Union


def read_graph6(path: str, output_format: str = "networkit") -> Generator[nk.Graph, None, None]:
    
    output_mapper = lambda x: nk.nxadapter.nx2nk(x) if output_format == "networkit" else x

    with open(path, 'r') as f:
        for line in map(str.strip, f):
            if not line:
                continue

            graph = nx.from_graph6_bytes(line.encode())
            graph = output_mapper(graph)
            yield graph
