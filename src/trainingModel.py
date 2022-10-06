import dgl
from dgl.data import DGLDataset
from dgl.nn import SAGEConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import time
import os
from sklearn.metrics import r2_score
from dgl.nn import GNNExplainer
import matplotlib.pyplot as plt

class KarateClubDataset(DGLDataset):
    def __init__(self, edges_data, nodes_data):
        self.edges_data = edges_data
        self.nodes_data = nodes_data
        super().__init__(name='network')
        
    def process(self):        
        node_features_1 = torch.from_numpy(self.nodes_data['port'].to_numpy())
        node_features_1 = torch.from_numpy(OneHotEncoder().fit_transform(node_features_1.reshape(-1, 1)).toarray()) 
        node_features_2 = torch.from_numpy(self.nodes_data['transport'].to_numpy())
        node_features_3 = torch.from_numpy(self.nodes_data['cloud'].to_numpy())
        node_features_4 = torch.from_numpy(self.nodes_data['vpn'].to_numpy())
        node_features_5 = torch.from_numpy(self.nodes_data['database'].to_numpy())
        node_features_6 = torch.from_numpy(self.nodes_data['devops'].to_numpy())
        node_features_7 = torch.from_numpy(self.nodes_data['honeypot'].to_numpy())
        node_features_8 = torch.from_numpy(self.nodes_data['self-signed'].to_numpy())
        node_features_9 = torch.from_numpy(self.nodes_data['etc'].to_numpy())
        node_features_10 = torch.from_numpy(self.nodes_data['info_os'].to_numpy())
        node_features_10 = torch.from_numpy(OneHotEncoder().fit_transform(node_features_10.reshape(-1, 1)).toarray()) 
        
        node_features = torch.stack([node_features_1[:,0], node_features_1[:,1],node_features_1[:,2],node_features_1[:,3],node_features_1[:,4],node_features_1[:,5],
        node_features_10[:,0],node_features_10[:,1],node_features_10[:,2],node_features_10[:,3],node_features_10[:,4], node_features_2, node_features_3, 
        node_features_4, node_features_5, node_features_6, node_features_7, node_features_8, node_features_9], 1) 

        edge_features = torch.from_numpy(self.edges_data['Weight'].to_numpy())
        edges_src = torch.from_numpy(self.edges_data['Src'].to_numpy())
        edges_dst = torch.from_numpy(self.edges_data['Dst'].to_numpy())
        
        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=self.nodes_data.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.edata['weight'] = edge_features
        
    def __getitem__(self, i):
        return self.graph
    
    def __len__(self):
        return 1

def make_graph_dataset(seed, num_of_nodes):
    assert os.path.exists("../graph_dataset")
    assert os.path.exists("../graph_data")
    node_data = pd.read_csv(f'../graph_data/DAG_nodes_features_{seed}_{num_of_nodes}.csv', index_col=[0])
    dataset = KarateClubDataset(pd.read_csv('../graph_dataset/DAG_train_edges.csv', index_col=[0]), node_data)
    train_g = dataset[0]
    KarateClubDataset(pd.read_csv('../graph_dataset/DAG_train_normal_edges.csv', index_col=[0]), node_data)
    train_nor_g = dataset[0]
    KarateClubDataset(pd.read_csv('../graph_dataset/DAG_train_vulns_edges.csv', index_col=[0]), node_data)
    train_vul_g = dataset[0]
    dataset = KarateClubDataset(pd.read_csv('../graph_dataset/DAG_test_edges.csv', index_col=[0]), node_data)
    test_g = dataset[0]
    KarateClubDataset(pd.read_csv('../graph_dataset/DAG_test_normal_edges.csv', index_col=[0]), node_data)
    test_nor_g = dataset[0]
    KarateClubDataset(pd.read_csv('../graph_dataset/DAG_test_vulns_edges.csv', index_col=[0]), node_data)
    test_vul_g = dataset[0]
    return train_g, train_nor_g, train_vul_g, test_g, test_nor_g, test_vul_g

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv1_5_1 = SAGEConv(h_feats, 128, 'mean')
        self.conv1_5_2 = SAGEConv(128, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
    
    def forward(self, graph, feat, eweight=None):
        h = self.conv1(graph, feat)
        h = F.relu(h)
        h = self.conv1_5_1(graph, h)
        h = F.relu(h)
        h = self.conv1_5_2(graph, h)
        h = F.relu(h)
        h = self.conv2(graph, h) 
        return h

class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': F.sigmoid(self.W2(F.relu(self.W1(h)))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

def compute_loss(pos_score, neg_score, train_vul_g, train_nor_g):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([train_vul_g.edata['weight'], train_nor_g.edata['weight']])
    return F.mse_loss(scores.float(), labels.float())

def compute_auc(pos_score, neg_score, test_vul_g, test_nor_g):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat([test_vul_g.edata['weight'], test_nor_g.edata['weight']]).numpy()
    return r2_score(labels, scores), scores
    
def print_plot(features_col, feat_mask):
    plt.style.use('seaborn')
    plt.figure()
    plt.barh([features_col[i] for i in range(len(features_col))], feat_mask)
    name = "Feature Importance"
    plt.title(name)
    plt.xlabel("Prabability")
    os.makedirs("../image", exist_ok=True)
    plt.savefig(f"../image/{name}.png", bbox_inches='tight')

def training(train_g, train_nor_g, train_vul_g, test_vul_g, test_nor_g):
    train_g.ndata['feat'] = torch.nan_to_num(train_g.ndata['feat'])
    a = train_g.ndata['feat']
    for x in np.isnan(a.numpy()):
        for y in x:
            if y:
                print("s")

    model = GraphSAGE(train_g.ndata['feat'].shape[1], 32)
    pred = MLPPredictor(32)
    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)
    if os.path.exists("../model/cvss_prediction_graphsage.pt") and os.path.exists("../model/cvss_prediction_mlp.pt"):
        model.load_state_dict(torch.load("../model/cvss_prediction_graphsage.pt"))
        h = model(train_g, train_g.ndata['feat'].float())
        pred.load_state_dict(torch.load("../model/cvss_prediction_mlp.pt"), strict=False)

    else:
        print("[INFO] Training Start!")
        start = time.time()
        for e in range(100):
            h = model(train_g, train_g.ndata['feat'].float())
            pos_score = pred(train_vul_g, h)
            neg_score = pred(train_nor_g, h)
            loss = compute_loss(pos_score, neg_score, train_vul_g, train_nor_g)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print('In epoch {}, loss: {}'.format(e, loss))

        print(f'time: {time.time()-start}')
        print("[INFO] Training Done..")

        os.makedirs("../model", exist_ok=True)
        torch.save(model.state_dict(), "../model/cvss_prediction_graphsage.pt")
        torch.save(pred.state_dict(), "../model/cvss_prediction_mlp.pt")

    with torch.no_grad():
        pos_score = pred(test_vul_g, h)
        neg_score = pred(test_nor_g, h)
        start = time.time()
        r2score, Pred_Score = compute_auc(pos_score, neg_score, test_vul_g, test_nor_g)

    os.makedirs("../results", exist_ok=True)
    with open("../results/results.txt", "w") as file:
        file.write("----------------Result------------------\n")
        file.write(f'r2_score: {r2score}\n') 
        file.write(f'time(sec): {time.time()-start}\n')
        file.write("----------------------------------------")

    features = test_g.ndata['feat'].float()
    features = torch.nan_to_num(features)

    a = features
    for x in np.isnan(a.numpy()):
        for y in x:
            if y:
                print("s")

    explainer = GNNExplainer(model, num_hops=1)
    feat_mask, edge_mask = explainer.explain_graph(test_g, features)

    features_col = ["other_ports", "likely_http_ports", "likely_ssl_ports", "likely_ssh_ports", "low_open_freq", "high_open_freq",
                    "red_hat_linux", "cent_os", "linux", "etc_os", "none_os", "transport", "cloud", "vpn", "database", "devops", "honeypot", "self-signed", "etc"]
    print_plot(features_col, feat_mask)

    test_src_edges = test_vul_g.edges()[0].tolist() + test_nor_g.edges()[0].tolist()
    test_des_edges = test_vul_g.edges()[1].tolist() + test_nor_g.edges()[1].tolist()

    Res_score = pd.DataFrame({"Src":test_src_edges,
              "Dst": test_des_edges,
              "Weight": Pred_Score})
    Res_score.to_csv(f"../results/Res_score_{seed}.csv")

if __name__ == "__main__":
    seed = 7
    num_of_nodes = 3842
    train_g, train_nor_g, train_vul_g, test_g, test_nor_g, test_vul_g = make_graph_dataset(seed, num_of_nodes)
    training(train_g, train_nor_g, train_vul_g, test_vul_g, test_nor_g)
