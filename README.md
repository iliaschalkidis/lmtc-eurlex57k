## Large-Scale Multi-Label Text Classification on EU Legislation

This is the code used for the experiments described in the following paper:


> I. Chalkidis, M. Fergadiotis, P. Malakasiotis and I. Androutsopoulos, "Large-Scale Multi-Label Text Classification on EU Legislation". Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019), Florence, Italy, (short papers), 2019 (https://www.aclweb.org/anthology/P19-1636/)

## Requirements:

* \>= Python 3.6
* >= TensorFlow 2.1
* == TensorFlow-Hub 0.7.0
* \>= Gensim 3.5.0
* == Keras 2.2.4
* \>= NLTK 3.4
* \>= Scikit-Learn 0.20.1
* \>= Spacy 2.1.0
* \>= TQDM 4.28.1

## Quick start:

### Install python requirements:

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python
>>> import nltk
>>> nltk.download('punkt')
```

### Get pre-trained word embeddings (GloVe + Law2Vec):

```
wget -P data/vectors/ http://nlp.stanford.edu/data/glove.6B.zip
unzip -j data/vectors/glove.6B.zip data/vectors/glove.6B.200d.txt
echo -e "400000 200\n$(cat data/vectors/glove.6B.200d.txt)" > data/vectors/glove.6B.200d.txt
wget -O data/vectors/law2vec.200d.txt https://archive.org/download/Law2Vec/Law2Vec.200d.txt
```

### Download dataset (EURLEX57K):

```
wget -O data/datasets/datasets.zip http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/datasets.zip
unzip data/datasets/datasets.zip -d data/datasets/EURLEX57K
rm data/datasets/datasets.zip
rm -rf data/datasets/EURLEX57K/__MACOSX
mv data/datasets/EURLEX57K/dataset/* data/datasets/EURLEX57K/
rm -rf data/datasets/EURLEX57K/dataset
wget -O data/datasets/EURLEX57K/EURLEX57K.json http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/eurovoc_en.json
```

### Select training options from the configuration JSON file:

E.g., run a Label-wise Attention Network with BIGRUs (BIGRU-LWAN) with the best reported hyper-parameters

```
nano ltmc_configuration.json

{
  "task": {
    "dataset": "EURLEX57K",
    "decision_type": "multi_label"
  },
  "model": {
    "architecture": "LABEL_WISE_ATTENTION_NETWORK",
    "document_encoder": "grus",
    "n_hidden_layers": 1,
    "hidden_units_size": 300,
    "dropout_rate": 0.4,
    "word_dropout_rate": 0.00,
    "lr": 0.001,
    "batch_size": 16,
    "epochs": 50,
    "attention_mechanism": "attention",
    "token_encoding": "word2vec",
    "embeddings": "law2vec.200d.txt",
    "bert": "bertbase"
  },
  "sampling": {
    "max_sequences_size": null,
    "max_sequence_size": 5000,
    "max_label_size": 15,
    "few_threshold": 50,
    "hierarchical": false,
    "evaluation@k": 10
  }
}
```

**Supported models:** BIGRUS, LABEL_WISE_ATTENTION_NETWORK, ZERO_LABEL_WISE_ATTENTION_NETWORK, BERT
**Supported token encodings:** word2vec, elmo 
**Supported document encoders:** grus, cnns

### Train a model:

```
python lmtc.py
```
