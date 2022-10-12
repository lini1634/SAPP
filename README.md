# SAPP: Scalable Attack Path Prediction using Graph Neural Network
![AttackPathPrediction drawio](https://user-images.githubusercontent.com/44723287/194034358-829e8823-8237-4c91-8120-bbb7db188855.png)

## Abstract 
![abstract](https://user-images.githubusercontent.com/44723287/194035556-f318c2c3-020b-41f2-9c6a-3d0075b7c63e.png)

## Description of Directory
* data: parsing shodan file
* graph_data: output of *makeDAG.py* and *makeFeatureVec.py* in *src* directory
* graph_dataset: output of *makeDataset.py* in *src* directory
* image: image output of program
* model: model of GNN and MLP used predicting CVSS saved
* src: source code
* results: predicted CVSS in graph dataset format, r2 score, and inference time

## Description of src directory  
1) makeDAG.py: Data collection (in above figure)
2) makeFeatureVec.py: Make Dataset in Modeling (in above figure)
3) makeDataset.py: Make Training Dataset 
4) trainingModel.py: Training and Test based on GNN and MLP
5) pathGenerator.py: Find Attack Path

## Requirement (Name                 Version)
dgl-cuda11.3              0.8.1  
torch                        1.9.1  
  
## Run & Results
### 1-1) Run
```python3
python makeDAG.py
```
### 1-2) Results
Output files: 
* graph_data/DAG_nodes_{seed}\_{num_of_nodes}.csv
* graph_data/DAG_edges_{seed}\_{num_of_nodes}.csv

### 2-1) Run
```python3
python makeFeatureVec.py
```
### 2-2) Results
Output files: graph_data/DAG_nodes_features_{seed}\_{num_of_nodes}.csv

### 3-1) Run
```python3
python makeDataset.py
```
### 3-2) Results
Output files:
* graph_dataset/DAG_train_edges.csv
* graph_dataset/DAG_train_normal_edges.csv
* graph_dataset/DAG_train_vulns_edges.csv
* graph_dataset/DAG_test_edges.csv
* graph_dataset/DAG_test_normal_edges.csv
* graph_dataset/DAG_test_vulns_edges.csv

### 4-1) Run
```python3
python trainingModel.py
```
### 4-2) Results
Ouput files: 
* results/results.txt
```
----------------Result------------------
r2_score: 0.9102470316643958
time: 0.0319979190826416
----------------------------------------
```
* results/Res_score_{seed}.csv

### 5-1) Run
```python3
python pathGenerator.py
```
### 5-2) Results
Ouput files: results/results_path.txt
```  
*****Picked paths (end vertex is not specified)*****
[[2289, 1576, 1141], [2289, 1246, 1141], [2289, 1353, 1141], [2289, 1982, 1141], [2289, 1535, 1141], [2289, 897, 150], [2289, 1749, 150], [2289, 1030, 150], [2289, 834, 150], [2289, 683, 150], [2289, 1750, 452], [2289, 1681, 452], [2289, 1948, 452], [2289, 807, 452], [2289, 1960, 452], [2289, 1130, 1119], [2289, 1689, 1119], [2289, 2159, 1119], [2289, 2061, 1119], [2289, 1882, 1808, 1273, 776]]

*****Candidate paths (end vertex specified)*****
[[2289, 1889, 1239, 1141], [2289, 1141], [2289, 1889, 1576, 1141], [2289, 1492, 1353, 1141], [2289, 1841, 1213, 1211, 1141], [2289, 2046, 1199, 1141], [2289, 1618, 1262, 1141], [2289, 1860, 1625, 1141], [2289, 2046, 1141], [2289, 1903, 1751, 1141], [2289, 2226, 1141], [2289, 2064, 1934, 1141], [2289, 2180, 1141], [2289, 2217, 1653, 1141], [2289, 1199, 1141], [2289, 1860, 1504, 1141], [2289, 1570, 1386, 1141], [2289, 2159, 1653, 1141], [2289, 2287, 2061, 1141], [2289, 1690, 1141]]

*****Dijkstra*****
Dist: 0.012589648
Path: [2289, 1141]
dijkstra excution time (sec): 0.258939266204834

*****Astar*****
[2289, 1141]
astar excution time (sec): 0.0397341251373291
```

## Publications
```
SAPP: Scalable Attack Path Prediction using Graph Neural Network
@article{
TBD
}
```

## About
This program is authored and maintained by **Haerin Kim**, and **Jeong Do Yoo**.  
> Email: rlagoflszz@gmail.com, opteryx25104@korea.ac.kr
