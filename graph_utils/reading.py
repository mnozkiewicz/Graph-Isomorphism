from typing import Generator

import networkit as nk
import networkx as nx


def read_graph6(
    path: str, output_format: str = "networkit"
) -> Generator[nk.Graph, None, None]:

    def output_mapper(graph: nx.Graph):
        return nk.nxadapter.nx2nk(graph) if output_format == "networkit" else graph

    with open(path, "r") as f:
        for line in map(str.strip, f):
            if not line:
                continue

            graph = nx.from_graph6_bytes(line.encode())
            graph = output_mapper(graph)
            yield graph
