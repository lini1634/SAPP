import pandas as pd
import numpy as np
from dijkstra import dijkstra
from astar import astar
import time
import os 

import random
import copy

random.seed(777)


def print_log(target, log_container: list = None):
    if log_container is not None:
        log_container.append(str(target) + '\n')
    else:
        print(target)


def load_edge_data(edge_data_file_path: str) -> pd.DataFrame:
    # print(edge_data_file_path)
    edge_data = pd.read_csv(edge_data_file_path)
    return edge_data[['Src', 'Dst', 'Weight']]


def convert_edge_format(edge_data: pd.DataFrame) -> dict:
    result = {}
    edge_data_arr = edge_data.to_numpy()
    for elem in edge_data_arr:
        weight = elem[2]
        if weight == 0.0:
            continue
        src_node = int(elem[0])
        dst_node = int(elem[1])
        if src_node not in result:
            result[src_node] = []
        result[src_node].append(dst_node)

    return result


def extract_paths(edge_data: dict, max_path_len: int, init_node: int = -1) -> (list, int):
    selected_paths = []
    if init_node == -1:
        random_init_node = random.sample(list(edge_data.keys()), 1)[0]
    else:
        random_init_node = init_node

    def _search_dfs_graph(_curr_path, _depth):
        if _depth >= max_path_len:
            selected_paths.append(_curr_path)
            return

        _curr_node = _curr_path[-1]
        if _curr_node not in edge_data or len(edge_data[_curr_node]) == 0:
            selected_paths.append(_curr_path)
            return

        _next_node_indices = random.choices(edge_data[_curr_path[-1]], k=10)
        for _next_node in _next_node_indices:
            if _next_node in _curr_path:
                continue

            _next_path = copy.deepcopy(_curr_path)
            _next_path.append(_next_node)
            _search_dfs_graph(_next_path, _depth + 1)
        return

    iter_counter = 0
    while True:
        _search_dfs_graph([random_init_node], 1)

        if len(selected_paths) >= 20000:
            break

        iter_counter += 1
        if iter_counter >= 2000:
            break

    # random_selected_paths = random.choices(selected_paths, k=20)
    return selected_paths, random_init_node


def select_path(path_list: list, start_node: int, n_paths: int, _end_node: int = -1) -> list:
    path_dict = {}
    for path in path_list:
        imm_path = []
        for idx, node in enumerate(path):
            if idx == 0:
                assert node == start_node
                continue

            if node not in path_dict:
                path_dict[node] = []

            if imm_path not in path_dict[node]:
                assert node not in imm_path
                _imm_path = copy.deepcopy(imm_path)
                path_dict[node].append(_imm_path)
            imm_path.append(node)

    error_counter = 0
    selected_paths = []
    while True:
        if _end_node == -1:
            end_node_list = list(path_dict.keys())
            end_node = random.sample(end_node_list, k=1)[0]
        else:
            end_node = _end_node

        if end_node not in path_dict.keys():    # Exception case
            return []

        if len(path_dict[end_node]) >= 5:
            n_samples = 5
        else:
            n_samples = len(path_dict[end_node])
        for imm_path in random.sample(path_dict[end_node], k=n_samples):
            selected_paths.append([start_node] + imm_path + [end_node])

        if len(selected_paths) >= n_paths:
            return selected_paths[:n_paths]

        error_counter += 1
        if error_counter >= 1000:
            # print(f'Proper path does not exist.')
            for imm_path in random.sample(path_dict[end_node], k=len(path_dict[end_node])):
                selected_paths.append([start_node] + imm_path + [end_node])

            return selected_paths


def select_paths(start_vertex: int, end_vertex: int, n_paths: int, edge_data_path: str) -> list:
    edge_data = load_edge_data(edge_data_path)
    edge_data = convert_edge_format(edge_data)

    result = []
    force_stop_counter = 0
    while True:
        max_path_len = random.randint(2, 6)     # At most 7 is recommended
        extracted_paths, _ = extract_paths(edge_data, max_path_len=max_path_len, init_node=start_vertex)
        selected_paths = select_path(extracted_paths, start_vertex, n_paths, _end_node=end_vertex)

        for selected_path in selected_paths[:random.randint(3, n_paths)]:
            if selected_path not in result:
                result.append(selected_path)

        if len(result) >= n_paths:
            break

        force_stop_counter += 1
        if force_stop_counter >= 100:
            break

    return result[:n_paths]


def pick_random_paths(start_vertex: int, n_paths: int, edge_data_path: str) -> list:
    return select_paths(start_vertex, -1, n_paths, edge_data_path)


class DirectedGraph:
    def __init__(self, num_of_vertices):
        self.v = num_of_vertices
        self.edges = [[-1 for _ in range(num_of_vertices)] for _ in range(num_of_vertices)]
        self.visited = []

    def add_edge(self, u, v, weight):
        self.edges[u][v] = weight


def build_graph(node_data_path, edge_data_path):
    node_df = pd.read_csv(node_data_path)
    node_df = node_df[['Id']]
    graph = DirectedGraph(len(node_df))
    edge_df = pd.read_csv(edge_data_path)
    edge_df = edge_df[['Src', 'Dst', 'Weight']]
    for _src, _dst, weight in np.array(edge_df):
        src = int(_src)
        dst = int(_dst)
        graph.add_edge(src, dst, weight)
    return graph


def run_algorithm(node_data_path, edge_data_path, start_vertex, end_vertex, log_str: list = None) -> None:
    graph = build_graph(node_data_path, edge_data_path)

    print_log("*****Dijkstra*****", log_str)
    start = time.time()
    dist = dijkstra(graph, start_vertex)
    end = time.time()
    for vertex in dist:
        if len(dist[vertex]["path"]) > 1:
            if dist[vertex]["path"][-1] == end_vertex:
                print_log(f'Dist: {dist[vertex]["distance"]}', log_str)
                print_log(f'Path: {dist[vertex]["path"]}', log_str)
    print_log(f"dijkstra excution time (sec): {end - start}", log_str)
    print_log('', log_str)

    print_log("*****Astar*****", log_str)
    start = time.time()
    path = astar(graph, start_vertex, end_vertex)
    end = time.time()
    print_log(path, log_str)
    print_log(f"astar excution time (sec): {end - start}", log_str)


if __name__ == '__main__':
    seed = 7
    num_of_nodes = 3842
    log_str_ = []

    assert os.path.exists("../graph_data")
    assert os.path.exists("../results")
    NODE_DATA_PATH = f'../graph_data/DAG_nodes_features_{seed}_{num_of_nodes}.csv'
    EDGE_DATA_PATH = f'../results/Res_score_{seed}.csv'

    start_vertex_ = 2289
    n_paths_ = 20
    picked_paths_ = pick_random_paths(start_vertex_, n_paths_, EDGE_DATA_PATH)
    print_log('*****Picked paths (end vertex is not specified)*****', log_str_)
    print_log(picked_paths_, log_str_)
    print_log('', log_str_)

    # It is recommended that end_vertex_ is selected from the end node of one of the picked_paths_ above.
    end_vertex_ = 1141
    selected_paths_ = select_paths(start_vertex_, end_vertex_, n_paths_, EDGE_DATA_PATH)
    print_log('*****Candidate paths (end vertex specified)*****', log_str_)
    print_log(selected_paths_, log_str_)
    print_log('', log_str_)

    run_algorithm(NODE_DATA_PATH, EDGE_DATA_PATH, start_vertex_, end_vertex_, log_str_)

    if log_str_ is not None:
        with open('../results/results_path.txt', 'w') as txt_file:
            txt_file.writelines(log_str_)
