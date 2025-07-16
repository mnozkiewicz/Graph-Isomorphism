import networkx as nx
import numpy as np
import networkit as nk
from graph_utils import visualisation
import random
import os
from typing import Dict, List, Any, Tuple, Callable, Generator
import json
import pandas as pd
import functools
from graph_utils.reading import read_graph6

from descriptors.embeddings import create_embedding_function, normalize_features
import xxhash

READ_PATH = 'raw_datasets'
SAVING_PATH = 'processed_datasets'


ORDER = ['number_of_nodes', 'features', 'include_node_features', 'normalize']

def open_test_enviroment(func):
    '''Decorator for doing basic reading from the file + preparing embedding function. The `func` parameter must take graph_reader and `embedding_function`'''
    @functools.wraps(func)
    def wrapper(number_of_nodes, features, **function_kwargs):

        file_path = os.path.join(READ_PATH, f"graph{number_of_nodes}c.g6")
        graph_reader = read_graph6(file_path) 
        embedding_function = create_embedding_function(features, 
                                                bins_per_feature=number_of_nodes**2, 
                                                **function_kwargs
                                                )
        return func(graph_reader, embedding_function)

    return wrapper

@open_test_enviroment
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

@open_test_enviroment
def find_optimal_histogram_ranges(graph_reader, embedding_function: Callable) ->List[Tuple[int, int]]:

    first_graph = next(graph_reader)
    function_values: np.ndarray= embedding_function(first_graph)
    hist_ranges = [(min(histogram), max(histogram)) for histogram in function_values]

    for graph_id, graph in enumerate(graph_reader, start=1):
        if graph_id % 2000 == 0:
            print(graph_id)
        function_values = embedding_function(graph)
        hist_ranges = [(min(ranges[0], min(values)), max(ranges[1], min(values))) for ranges, values in zip(hist_ranges, function_values)]

    
    return hist_ranges



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

    # reading outputs file, having parameters values and list of all 
    file_path = os.path.join(SAVING_PATH, 'table.parquet')
    if os.path.exists(file_path):
        outputs_df = pd.read_parquet(file_path)
    else:
        outputs_df = pd.DataFrame(columns=ORDER + ['result'])

    try:
        for kwargs in arguments_lists:

            # check if this set of parameters already was run
            exists = outputs_df.apply(lambda row: _row_matches(row, {key : kwargs[key] for key in ORDER}), axis=1).any()
            if exists:
                print(f'test with {kwargs} parameters existed, skipped')
                continue
            
            # TODO check if this histogram ranges was calculated before
            kwargs['embeddings'] = False
            histogram_ranges = find_optimal_histogram_ranges(**kwargs)
            print(histogram_ranges)

            kwargs['embeddings'] = True
            kwargs['histogram_ranges'] = histogram_ranges

            result = select_problematic_ids(**kwargs)
            result = [item for sublist in result.values() for item in sublist] if result else np.array([-1])
            print(result)

            outputs_df = pd.concat([outputs_df, pd.DataFrame([dict(**{key : kwargs[key] for key in ORDER}, result=result)])]) 
    
    except KeyboardInterrupt:
        pass

    finally:
        outputs_df.to_parquet(file_path)

    



    