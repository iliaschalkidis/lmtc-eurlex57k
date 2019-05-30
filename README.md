## Large-Scale Multi-Label Text Classification on EU Legislation

This is the code used for the experiments described in the following paper:


> I. Chalkidis, M. Fergadiotis, P. Malakasiotis and I. Androutsopoulos, "Large-Scale Multi-Label Text Classification on EU Legislation". Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019), Florence, Italy, (short papers), 2019.

### Requirements:

* >= Python 3.6
* >= TensorFlow 1.13

# Quick start:


### Get pre-trained word embeddings (GloVe):

```
wget -P data/vectors/word2vec http://nlp.stanford.edu/data/glove.6B.zip
unzip -j data/vectors/word2vec/glove.6B.zip glove.6B.200d.txt
```

### Download dataset (EURLEX57K):

```
wget -P data/datasets http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K
```

### Train a model:

```
python train_model.py
```
