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
```
cmd에서 코드 실행 결과..
```
### 1-3) Ouput files

### 2-1) Run
```python3
python makeFeatureVec.py
```
### 2-2) Results
```
cmd에서 코드 실행 결과..
```
### 2-3) Ouput files

### 3-1) Run
```python3
python makeDataset.py
```
### 3-2) Results
```
cmd에서 코드 실행 결과..
```
### 3-3) Ouput files

### 4-1) Run
```python3
python trainingModel.py
```
### 4-2) Results
```
cmd에서 코드 실행 결과..
```
### 4-3) Ouput files

### 5-1) Run
```python3
python pathGenerator.py
```
### 5-2) Results
```
cmd에서 코드 실행 결과..
```
### 5-3) Ouput files

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
