## Large-Scale Multi-Label Text Classification on EU Legislation

___
**NOTICE:** Moving and tidying up from https://github.com/AnonymousXMTC2019/LMTC that was used in the anonymity period.  
___
This is the code used for the experiments described in the following paper:


> I. Chalkidis, M. Fergadiotis, P. Malakasiotis and I. Androutsopoulos, "Large-Scale Multi-Label Text Classification on EU Legislation". Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019), Florence, Italy, (short papers), 2019.

## Requirements:

* >= Python 3.6
* >= TensorFlow 1.13

## Quick start:


### Get pre-trained word embeddings (GloVe + Law2Vec):

```
wget -P data/vectors/word2vec http://nlp.stanford.edu/data/glove.6B.zip
unzip -j data/vectors/word2vec/glove.6B.zip glove.6B.200d.txt
wget -P data/vectors/word2vec https://archive.org/download/Law2Vec/Law2Vec.200d.txt
```

### Download dataset (EURLEX57K):

```
wget -P data/datasets http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K
```

### Select training options in configuration json file

E.g., run a Label-wise Attention Network with BIGRUs (BIGRU-LWAN) with the best reported hyper-parameters

```
nano ltmc_configuration.json

{
  "task": {
    "dataset": "eurovoc_en",
    "decision_type": "multi_label"
  },
  "model": {
    "architecture": "LABEL_WISE_ATTENTION_NETWORK",
    "document_encoder": "grus",
    "label_encoder": null,
    "n_hidden_layers": 1,
    "hidden_units_size": 300,
    "dropout_rate": 0.4,
    "word_dropout_rate": 0.00,
    "lr": 0.001,
    "batch_size": 16,
    "epochs": 2,
    "attention_mechanism": "attention",
    "token_encoding": "word2vec",
    "embeddings": "en/glove.6B.200d.bin",
  },
  "sampling": {
    "max_sequences_size": null,
    "max_sequence_size": 5000,
    "hierarchical": false,
    "evaluation@k": 10
  }
}
```

### Train a model:

```
python run_lmtc.py
```
