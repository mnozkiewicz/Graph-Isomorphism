import networkx as nx
import numpy as np
import networkit as nk
from graph_utils import visualisation
import random
import os
from typing import Dict, List, Any, Tuple
import json

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


def tests(arguments_lists: List[Dict[str: Any]]):
    file_path = os.path.join(SAVING_PATH, 'table')
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {}

    for kwargs in arguments_lists:
        current_dict = data

        for i, key in enumerate(ORDER[:-1]):
            if kwargs[key] in current_dict:
                current_dict = current_dict[kwargs[key]]
            else:
                for added_key in ORDER[i:-1]:
                    current_dict[kwargs[added_key]] = {}
                    current_dict = current_dict[kwargs[added_key]]
        else:
            if kwargs[ORDER[-1]] in current_dict:
                continue


        current_dict[kwargs[ORDER[-1]]] = single_test_collisions(**kwargs)



    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    



    