import functools
import json
import os
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import xxhash

from descriptors.embeddings import create_embedding_function, normalize_features
from graph_utils.reading import read_graph6

READ_PATH = "raw_datasets"
SAVING_PATH = "processed_datasets"


ORDER = ["number_of_nodes", "features", "normalize"]


def open_test_enviroment(func):
    """Decorator for doing basic reading from the file + preparing embedding function. The `func` parameter must take graph_reader and `embedding_function`"""

    @functools.wraps(func)
    def wrapper(number_of_nodes, features, **function_kwargs):

        file_path = os.path.join(READ_PATH, f"graph{number_of_nodes}c.g6")
        graph_reader = read_graph6(file_path)
        embedding_function = create_embedding_function(
            features, bins_per_feature=number_of_nodes**2, **function_kwargs
        )
        return func(graph_reader, embedding_function)

    return wrapper


@open_test_enviroment
def select_problematic_ids(graph_reader, embedding_function) -> Dict[str, list[int]]:
    """goes through all the graphs and selects only the ones that have collisions on embedding"""

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
def find_optimal_histogram_ranges(
    graph_reader, embedding_function: Callable
) -> List[Tuple[int, int]]:
    """function that goes through all descriptor values per graphs and finds minimum and maximum of each feature value"""
    first_graph = next(graph_reader)
    function_values: np.ndarray = embedding_function(first_graph)
    hist_ranges = [(min(histogram), max(histogram)) for histogram in function_values]

    for graph_id, graph in enumerate(graph_reader, start=1):
        if graph_id % 2000 == 0:
            print(graph_id)
        function_values = embedding_function(graph)
        hist_ranges = [
            (min(ranges[0], min(values)), max(ranges[1], min(values)))
            for ranges, values in zip(hist_ranges, function_values)
        ]

    return hist_ranges


def reduce_number_of_features(
    stored_ranges_dict, number_of_nodes, features, normalize, **other_features
):
    if number_of_nodes not in stored_ranges_dict:
        stored_ranges_dict[number_of_nodes] = {normalize: {}}
        return features

    if normalize not in stored_ranges_dict[number_of_nodes]:
        stored_ranges_dict[number_of_nodes][normalize] = {}
        return features

    features_to_be_used = [
        feature
        for feature in features
        if feature not in stored_ranges_dict[number_of_nodes][normalize]
    ]
    return features_to_be_used


def update_histogram_ranges(
    stored_ranges_dict,
    features_to_update,
    histogram_ranges,
    features,
    number_of_nodes,
    normalize,
    **other_features,
) -> List[Tuple[int, int]]:
    for feature, ranges in zip(features_to_update, histogram_ranges):
        stored_ranges_dict[number_of_nodes][normalize][feature] = tuple(
            map(float, ranges)
        )

    return [
        stored_ranges_dict[number_of_nodes][normalize][feature] for feature in features
    ]


def _values_equal(a, b):
    if isinstance(a, (np.ndarray, list)) and isinstance(b, (np.ndarray, list)):
        return np.array_equal(a, b)
    return a == b


def _row_matches(row, criteria):
    return all(_values_equal(row[k], v) for k, v in criteria.items())


def convert_bool_from_str(str: str):
    if str in ["true", "True"]:
        return True
    elif str in ["false", "False"]:
        return False
    raise ValueError


def tests(arguments_lists: List[Dict[str, Any]]):

    # reading outputs file, having parameters values and list of all
    output_path = os.path.join(SAVING_PATH, "table.parquet")
    if os.path.exists(output_path):
        outputs_df = pd.read_parquet(output_path)
    else:
        outputs_df = pd.DataFrame(columns=ORDER + ["result"])

    histograms_path = os.path.join(SAVING_PATH, "histograms_ranges.json")
    if os.path.exists(histograms_path):
        with open(histograms_path, "r") as f:
            read_data = json.load(f)

        stored_histogram_ranges = {int(key): value for key, value in read_data.items()}
        for size_dict in stored_histogram_ranges.values():
            for k, v in list(size_dict.items()):
                size_dict[convert_bool_from_str(k)] = v
                del size_dict[k]
    else:
        stored_histogram_ranges = {}

    try:
        for kwargs_original in arguments_lists:
            kwargs = kwargs_original.copy()
            kwargs["features"] = normalize_features(kwargs["features"])
            # check if this set of parameters already was run
            exists = outputs_df.apply(
                lambda row: _row_matches(
                    row, {key: kwargs_original[key] for key in ORDER}
                ),
                axis=1,
            ).any()
            if len(outputs_df) > 0 and exists:
                print(
                    f"test with { {key : kwargs[key] for key in ORDER} } parameters existed, skipped"
                )
                continue

            # reusing already calculated histogram ranges
            features_to_be_used = reduce_number_of_features(
                stored_histogram_ranges, **kwargs
            )

            kwargs2 = kwargs.copy()
            kwargs2["features"] = features_to_be_used
            histogram_ranges = find_optimal_histogram_ranges(**kwargs2, embeddings=False)  # type: ignore

            histogram_ranges = update_histogram_ranges(
                stored_histogram_ranges, features_to_be_used, histogram_ranges, **kwargs
            )

            result = select_problematic_ids(**kwargs, embeddings=True, histogram_ranges=histogram_ranges)  # type: ignore
            result = (
                [item for sublist in result.values() for item in sublist]
                if result
                else np.array([-1])
            )
            print(result)
            print()

            outputs_df = pd.concat(
                [
                    outputs_df,
                    pd.DataFrame(
                        [
                            dict(
                                **{key: kwargs_original[key] for key in ORDER},
                                result=result,
                            )
                        ]
                    ),
                ]
            )

    except KeyboardInterrupt:
        pass

    finally:
        outputs_df.to_parquet(output_path)
        with open(histograms_path, "w") as f:
            json.dump(stored_histogram_ranges, f, indent=4)
