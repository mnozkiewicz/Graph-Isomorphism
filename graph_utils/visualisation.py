import networkx as nx
import networkit as nk
from typing import Union


def plot_graph(graph: Union[nx.Graph, nk.Graph]) -> None:
    if isinstance(graph, nk.Graph):
        graph = nk.nxadapter.nk2nx(graph)

    layout = nx.kamada_kawai_layout(graph)
    nx.draw(
        graph, 
        with_labels=True,
        node_color='lightgreen', 
        node_size=500, 
        font_size=10, 
        font_color='black', 
        pos=layout
    )
