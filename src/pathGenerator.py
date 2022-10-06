import pandas as pd
import numpy as np
from dijkstra import dijkstra
from astar import astar
import time
import os 

class DirectedGraph:
    def __init__(self, num_of_vertices):
        self.v = num_of_vertices
        self.edges = [[-1 for i in range(num_of_vertices)] for j in range(num_of_vertices)]
        self.visited = []
    def add_edge(self, u, v, weight):
        self.edges[u][v] = weight

def build_graph(NODE_DATA_PATH, EDGE_DATA_PATH):
    node_df = pd.read_csv(NODE_DATA_PATH)
    node_df = node_df[['Id']]
    graph = DirectedGraph(len(node_df))
    edge_df = pd.read_csv(EDGE_DATA_PATH)
    edge_df = edge_df[['Src', 'Dst', 'Weight']]
    for _src, _dst, weight in np.array(edge_df):
        src = int(_src)
        dst = int(_dst)
        graph.add_edge(src, dst, weight)
    return graph

def run_dijkstra(graph, start_vertex, end_vertex):
    print("*****Dijkstra*****")
    dist = dijkstra(graph, start_vertex)
    for vertex in dist:
        if len(dist[vertex]["path"]) > 1:
            if dist[vertex]["path"][-1] == end_vertex:
                print(f'Dist: {dist[vertex]["distance"]}')
                print(f'Path: {dist[vertex]["path"]}')

def run_astar(graph, start_vertex, end_vertex):
    print("*****Astar*****")
    path = astar(graph, start_vertex, end_vertex)
    print(path)

def run_algorithm(node_data_path, edge_data_path, start_vertex, end_vertex) -> None:
    graph = build_graph(node_data_path, edge_data_path)

    start = time.time()  
    run_dijkstra(graph, start_vertex, end_vertex)
    print("dijkstra excution time (sec):", time.time() - start) 

    start = time.time()  
    run_astar(graph, start_vertex, end_vertex)
    print("astar excution time (sec):", time.time() - start) 

if __name__ == '__main__':
    seed = 7
    num_of_nodes = 3842
    start_vertex = 1337
    end_vertex = 0
    assert os.path.exists("../graph_data")
    assert os.path.exists("../results")
    NODE_DATA_PATH = f'../graph_data/DAG_nodes_features_{seed}_{num_of_nodes}.csv'
    EDGE_DATA_PATH = f'../results/Res_score_{seed}.csv'
    run_algorithm(NODE_DATA_PATH, EDGE_DATA_PATH, start_vertex, end_vertex)
