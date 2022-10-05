import pandas as pd
from tqdm import tqdm
import os

def read_graph_data(edge_path):
    edges_data = pd.read_csv(edge_path, index_col=[0])
    return edges_data

def save_dataset(src_data, dst_data, label, path):
    df = pd.DataFrame({"Src": src_data,
                        "Dst": dst_data,
                        "Weight": label})
    df.to_csv(path)
    
def make_dataset(edges_data):
    arr_edges_data = edges_data.values
    normal_u, normal_v, normal_label, vulns_u, vulns_v, vulns_label = [], [], [], [], [], []
    for edge in tqdm(arr_edges_data):
        weight = edge[2]
        if weight == 0:
            normal_u.append(int(edge[0]))
            normal_v.append(int(edge[1]))
            normal_label.append(weight)
        else:
            vulns_u.append(int(edge[0]))
            vulns_v.append(int(edge[1]))
            vulns_label.append(weight)
    train_size = 0.8
    len_nor = len(normal_label)
    train_nor_u = normal_u[:int(len_nor*train_size)]
    test_nor_u = normal_u[int(len_nor*train_size):]
    train_nor_v = normal_v[:int(len_nor*train_size)]
    test_nor_v = normal_v[int(len_nor*train_size):]
    train_nor_label = normal_label[:int(len_nor*train_size)]
    test_nor_label = normal_label[int(len_nor*train_size):]
    len_vul = len(vulns_label)
    train_vul_u = vulns_u[:int(len_vul*train_size)]
    test_vul_u = vulns_u[int(len_vul*train_size):]
    train_vul_v = vulns_v[:int(len_vul*train_size)]
    test_vul_v = vulns_v[int(len_vul*train_size):]
    train_vul_label = vulns_label[:int(len_vul*train_size)]
    test_vul_label = vulns_label[int(len_vul*train_size):]

    os.makedirs("../graph_dataset", exist_ok=True)
    save_dataset(train_nor_u+train_vul_u, train_nor_v+train_vul_v, train_nor_label+train_vul_label, "../graph_dataset/DAG_train_edges.csv")
    save_dataset(train_nor_u, train_nor_v, train_nor_label, "../graph_dataset/DAG_train_normal_edges.csv")
    save_dataset(train_vul_u, train_vul_v, train_vul_label, "../graph_dataset/DAG_train_vulns_edges.csv")
    save_dataset(test_nor_u+test_vul_u, test_nor_v+test_vul_v, test_nor_label+test_vul_label, "../graph_dataset/DAG_test_edges.csv")
    save_dataset(test_nor_u, test_nor_v, test_nor_label, "../graph_dataset/DAG_test_normal_edges.csv")
    save_dataset(test_vul_u, test_vul_v, test_vul_label, "../graph_dataset/DAG_test_vulns_edges.csv")

def main(seed, num_of_nodes):
    edge_path = f'../graph_data/DAG_edges_{seed}_{num_of_nodes}.csv'
    edges_data = read_graph_data(edge_path)
    make_dataset(edges_data)
    
if __name__ == "__main__":
    seed = 7
    num_of_nodes = 3842
    main(seed, num_of_nodes)