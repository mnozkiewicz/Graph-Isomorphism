import networkx as nx
import numpy as np
import networkit as nk
from graph_utils import visualisation
import random
import os
from typing import Dict, List, Any, Tuple
import json
import pandas as pd

from graph_utils.reading import read_graph6

from descriptors.embeddings import create_embedding_function, normalize_features
import xxhash

READ_PATH = 'raw_datasets'
SAVING_PATH = 'processed_datasets'


ORDER = ['number_of_nodes', 'features', 'histogram_range', 'include_node_features', 'normalize']



def select_problematic_ids(graph_reader, embedding_function) -> Dict[str, list[int]]:
    
    collisions: dict[str, list[int]] = {}
    hashes: dict[str, int] = {}

    for graph_id, graph in enumerate(graph_reader):
        if graph_id % 2000 == 0:
            print(graph_id)

        embedding = embedding_function(graph)
        h = xxhash.xxh128_hexdigest(embedding.tobytes())

        if h not in hashes:
            hashes[h] = graph_id
        else:
            if h in collisions:
                collisions[h].append(graph_id)
            else:
                collisions[h] = [hashes[h], graph_id]

    return collisions


def single_test_collisions(number_of_nodes, features, **function_kwargs) -> Dict[str, list[int]]:

    file_path = os.path.join(READ_PATH, f"graph{number_of_nodes}c.g6")
    graph_reader = read_graph6(file_path) 
    
    embedding_function = create_embedding_function(features, 
                                              bins_per_feature=number_of_nodes**2, 
                                              **function_kwargs
                                              )
    
    return select_problematic_ids(graph_reader, embedding_function)

def _values_equal(a, b):
    if isinstance(a, (np.ndarray, list)) and isinstance(b, (np.ndarray, list)):
        return np.array_equal(a, b)
    return a == b

def _row_matches(row, criteria):
    # for k, v in criteria.items():
    #     print(k, v, row[k], v.__class__.__name__,  row[k].__class__.__name__)
    # print(list(_values_equal(row[k], v) for k, v in criteria.items()))
    return all(_values_equal(row[k], v) for k, v in criteria.items())

def tests(arguments_lists: List[Dict[str, Any]]):
    file_path = os.path.join(SAVING_PATH, 'table.parquet')
    if os.path.exists(file_path):
        outputs_df = pd.read_parquet(file_path)
    else:
        outputs_df = pd.DataFrame(columns=ORDER + ['result'])

    for kwargs in arguments_lists:

        exists = outputs_df.apply(lambda row: _row_matches(row, kwargs), axis=1).any()
        if exists:
            print(f'test with {kwargs} parameters existed, skipped')
            continue
        result = single_test_collisions(**kwargs)
        result = [item for sublist in result.values() for item in sublist] if result else np.array([-1])
        print(result)

        outputs_df = pd.concat([outputs_df, pd.DataFrame([dict(**kwargs, result=result)])])


    outputs_df.to_parquet(file_path)

    



    