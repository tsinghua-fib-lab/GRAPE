# Automorphic Equivalence-aware Graph Neural Network

This repository is the official implementation of [Automorphic Equivalence-aware Graph Neural Network](https://arxiv.org/abs/2011.04218). 

![alt text](https://github.com/tsinghua-fib-lab/GRAPE/blob/main/main.png?raw=true)

## Dependencies

This implementation depends on the following enviornment and packages:

```setup
python == 2.7
pytorch == 1.4
numpy == 1.17.2
scipy == 1.6.3
sklearn ==0.24.1
networkx == 2.5
h5py == 2.9.0
GPUtil ==1.4.0
setproctitle == 1.1.10
```

## Training and Evaluation

To train and evaluate the model(s) in the paper, run this command:

```train
python grape_model.py [gpu_module_id] 
```


```train
python genetic_search.py [gpu_module_id] [dataset_id] 
```


## Results

Our model achieves the following classification accuracy:


|  | Hamilton | Lehigh | Rochester | JHU | Amherst | Cora | Citeseer | Amazon |
| ------------------ |---------------- | -------------- | ---------------- | -------------- | ---------------- | -------------- | ---------------- | -------------- |
|  GRAPE  | 28.1% | 27.3% | 25.0% | 34.6% | 32.6% | 87.1% | 74.6% | 58.6% |

