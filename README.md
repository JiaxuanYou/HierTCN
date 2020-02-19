# Hierarchical Temporal Convolutional Networks for Dynamic Recommender Systems
This repository is a Tensorflow implementation of HierTCN for Dynamic Recommender Systems.

[Jiaxuan You](https://cs.stanford.edu/~jiaxuan/), Yichen Wang, Aditya Pal, Pong Eksombatchai, Chuck Rosenberg, Jure Leskovec, [Hierarchical Temporal Convolutional Networks for Dynamic Recommender Systems](https://arxiv.org/abs/1904.04381), The Web Conference 2019 (WWW-2019).

## Requirements
Tensorflow (tested on 1.6.0), numpy, pandas, pickle


## Dataset
Download [XING dataset](http://2016.recsyschallenge.com/), only `interactions.csv` is needed. Copy `interactions.csv` to `dataset/` folder.

If XING dataset is not available, you can email Jiaxuan You for the dataset for solely research use.

## Run the code
```bash
python run_xing.py
```

## Outputs
Each run will generate a folder in `./run`, which can be accessed via tensorboard. A summarized text file is also generated.