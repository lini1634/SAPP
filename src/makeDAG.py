import pandas as pd 
import json
import numpy as np
from tqdm import tqdm
import ast
import networkx as nx
import random
import matplotlib.pyplot as plt
import os

def make_graph(seed, num_of_nodes):
    G=nx.fast_gnp_random_graph(num_of_nodes,0.5,seed=seed, directed=True)
    DAG = nx.DiGraph([(u,v,{'weight':random.randint(-10,10)}) for (u,v) in G.edges() if u<v])
    print(f"Generated raw graph is DAG: {nx.is_directed_acyclic_graph(DAG)}")
    return G, DAG

def random_subgraph_img(DAG, num_of_nodes):
    H = DAG.subgraph([random.randrange(num_of_nodes) for _ in range(12) ])
    plt.figure(figsize=(12, 8))
    nx.draw_networkx(H, node_color="skyblue", node_size=1000)
    plt.savefig("../image/Random subgraph of DAG (node = 12).pdf")
    print('Save Random subgraph of DAG (node = 12)..')

def read_shodan_parsing(parsing_path):
    shodan_query_parsing = pd.read_csv(parsing_path, index_col=[0])
    return shodan_query_parsing

def add_topology_id(shodan_query_parsing, G):
    shodan_query_parsing["TopologyId"] = 0
    shodan_Id = 0 
    for host_node in list(G.nodes()):
        shodan_query_parsing.loc[shodan_Id,"TopologyId"] = host_node
        shodan_Id += 1
    return shodan_query_parsing

def make_graph_dataset(DAG, shodan_query_parsing, seed, num_of_nodes):
    connection = list(DAG.edges())
    len(connection)

    edges = []
    for connections in tqdm(connection):
        Dst = int(connections[0])
        Src = int(connections[1])
        Weight = shodan_query_parsing[shodan_query_parsing["TopologyId"]==Dst]["vulns"].values[0]
        if Weight != str(0):
            li_Weight = ast.literal_eval(Weight)
            f_li_Weight = [float(i) for i in li_Weight]
            Weight = np.max(f_li_Weight)
        Weight = int(Weight)
        edges.append([Src, Dst, Weight])

    df_edges = pd.DataFrame(edges, columns=["Src", "Dst", "Weight"])
    df_edges["Weight"] = df_edges["Weight"]/10

    filter_col = ["TopologyId", "port", "cpe23", "transport", "info", "tags", "vulns"]
    nodes = shodan_query_parsing[filter_col].rename(columns={'TopologyId': 'Id'})

    os.makedirs('../graph_data', exist_ok=True)
    nodes.to_csv(f"../graph_data/DAG_nodes_{seed}_{num_of_nodes}.csv")
    df_edges.to_csv(f"../graph_data/DAG_edges_{seed}_{num_of_nodes}.csv")

def main(seed):
    shodan_query_parsing = read_shodan_parsing("../data/shodan_query_parsing_3842.csv")
    G, DAG = make_graph(seed, len(shodan_query_parsing))
    shodan_query_parsing = add_topology_id(shodan_query_parsing, G)
    make_graph_dataset(DAG, shodan_query_parsing, seed, len(shodan_query_parsing))

if __name__ == "__main__":
    seed = 7
    main(seed)
