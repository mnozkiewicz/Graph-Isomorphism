import networkx as nx
from typing import Generator, List, Union


def read_graph6_batches(path: str, batch_size: int = 1) -> Generator[List[nx.Graph], None, None]:

    if batch_size <= 0:
         raise ValueError("Batch size needs to be positive")
    
    batch = []
    with open(path, 'r') as f:
        for line in map(str.strip, f):
            if not line:
                continue  # skip empty lines

            graph = nx.from_graph6_bytes(line.encode())
            batch.append(graph)
            if len(batch) >= batch_size:
                yield batch
                batch = []

        if len(batch) > 0:
            yield batch
