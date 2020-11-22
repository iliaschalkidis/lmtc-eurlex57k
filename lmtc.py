import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import pickle
import json
import re
import time
import tempfile
import glob
import tqdm
import pdb


import numpy as np
from tempfile import TemporaryFile
from copy import deepcopy
from collections import Counter
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.utils import Sequence
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score, precision_score, recall_score
import tensorflow as tf
from json_loader import JSONLoader
from vectorizer import W2VVectorizer, ELMoVectorizer, BERTVectorizer,HgBERTVectorizer
from data import DATA_SET_DIR, MODELS_DIR
from configuration import Configuration
from metrics import mean_recall_k, mean_precision_k, mean_ndcg_score, mean_rprecision_k
from neural_networks.lmtc_networks.document_classification import DocumentClassification
from neural_networks.lmtc_networks.label_driven_classification import LabelDrivenClassification
from keras import backend as K
from keras.models import load_model
from neural_networks.layers.bert import BERT

LOGGER = logging.getLogger(__name__)


class LMTC:

    def __init__(self):
        super().__init__()
        if 'elmo' in Configuration['model']['token_encoding']:
            self.vectorizer = ELMoVectorizer()
            self.vectorizer2 = W2VVectorizer(w2v_model=Configuration['model']['embeddings'])
        elif 'bert' == Configuration['model']['architecture'].lower():
            self.vectorizer = BERTVectorizer()
        elif 'legalbert' == Configuration['model']['architecture'].lower():
            self.vectorizer = HgBERTVectorizer()
        else:
            self.vectorizer = W2VVectorizer(w2v_model=Configuration['model']['embeddings'])
        self.load_label_descriptions()

    def load_label_descriptions(self):
        LOGGER.info('Load labels\' data')
        LOGGER.info('-------------------')

        # Load train dataset and count labels
        train_files = glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], 'train', '*.json'))
        train_counts = Counter()
        for filename in tqdm.tqdm(train_files):
            with open(filename) as file:
                data = json.load(file)
                for concept in data['concepts']:
                    train_counts[concept] += 1

        train_concepts = set(list(train_counts))

        frequent, few = [], []
        for i, (label, count) in enumerate(train_counts.items()):
            if count > Configuration['sampling']['few_threshold']:
                frequent.append(label)
            else:
                few.append(label)

        # Load dev/test datasets and count labels
        rest_files = glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], 'dev', '*.json'))
        rest_files += glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], 'test', '*.json'))
        rest_concepts = set()
        for filename in tqdm.tqdm(rest_files):
            with open(filename) as file:
                data = json.load(file)
                for concept in data['concepts']:
                    rest_concepts.add(concept)

        # Load label descriptors
        with open(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'],
                               '{}.json'.format(Configuration['task']['dataset']))) as file:
            data = json.load(file)
            none = set(data.keys())

        none = none.difference(train_concepts.union((rest_concepts)))
        parents = []
        for key, value in data.items():
            parents.extend(value['parents'])
        none = none.intersection(set(parents))

        # Compute zero-shot group
        zero = list(rest_concepts.difference(train_concepts))
        true_zero = deepcopy(zero)
        zero = zero + list(none)


        # Compute margins for frequent / few / zero groups
        self.label_ids = dict()
        self.margins = [(0, len(frequent)+len(few)+len(true_zero))]
        k = 0
        for group in [frequent, few, true_zero]:
            self.margins.append((k, k+len(group)))
            for concept in group:
                self.label_ids[concept] = k
                k += 1
        self.margins[-1] = (self.margins[-1][0], len(frequent)+len(few)+len(true_zero))


        # Compute label descriptors representations
        label_terms = []
        self.label_terms_text = []
        for i, (label, index) in enumerate(self.label_ids.items()):
            label_terms.append([token for token in word_tokenize(data[label]['label']) if re.search('[A-Za-z]', token)])
            self.label_terms_text.append(data[label]['label'])
        self.label_terms_ids = self.vectorizer.vectorize_inputs(label_terms,
                                                                max_sequence_size=Configuration['sampling']['max_label_size'],
                                                                features=['word'])

        # Eliminate labels out of scope (not in datasets)
        self.labels_cutoff = len(frequent) + len(few) + len(true_zero)

        self.label_terms_ids = self.label_terms_ids[:self.labels_cutoff]
        label_terms = label_terms[:self.labels_cutoff]

        LOGGER.info('Labels shape:    {}'.format(len(label_terms)))
        LOGGER.info('Frequent labels: {}'.format(len(frequent)))
        LOGGER.info('Few labels:      {}'.format(len(few)))
        LOGGER.info('Zero labels:     {}'.format(len(true_zero)))

    def load_dataset(self, dataset_name,model_type):
        """
        Load dataset and return list of documents
        :param dataset_name: the name of the dataset,model_type:BERT ou sth 
        :return: list of Document objects
        """
        filenames = glob.glob(os.path.join(DATA_SET_DIR, Configuration['task']['dataset'], dataset_name, '*.json'))
        loader = JSONLoader()

        documents = []
        for filename in tqdm.tqdm(sorted(filenames)):
            # pdb.set_trace()
            document=loader.read_file(filename)
            if model_type.lower()=="bert":
                # pdb.set_trace()
                document.tokens=document.tokens[:500]#512
            documents.append(document)

        return documents

    def process_dataset(self, documents):
        """
         Process dataset documents (samples) and create targets
         :param documents: list of Document objects
         :return: samples, targets
         """
        samples = []
        targets = []
        for document in documents:
            if Configuration['sampling']['hierarchical']:
                samples.append(document.sentences)
            else:
                # pdb.set_trace()
                samples.append(document.tokens)
            targets.append(document.tags)

        del documents
        return samples, targets

    def encode_dataset(self, sequences, tags):
        if Configuration['sampling']['hierarchical']:
            samples = np.zeros((len(sequences), Configuration['sampling']['max_sequences_size'],
                                Configuration['sampling']['max_sequence_size'],), dtype=np.int32)

            targets = np.zeros((len(sequences), len(self.label_ids)), dtype=np.int32)
            for i, (sub_sequences, document_tags) in enumerate(zip(sequences, tags)):
                sample = self.vectorizer.vectorize_inputs(sequences=sub_sequences[:Configuration['sampling']['max_sequences_size']],
                                                          max_sequence_size=Configuration['sampling']['max_sequence_size'])
                samples[i, :len(sample)] = sample
                for tag in document_tags:
                    if tag in self.label_ids:
                        targets[i][self.label_ids[tag]] = 1
            samples = np.asarray(samples)
        else:
            samples = self.vectorizer.vectorize_inputs(sequences=sequences,
                                                       max_sequence_size=Configuration['sampling']['max_sequence_size'])

            if 'elmo' in Configuration['model']['token_encoding']:
                samples2 = self.vectorizer2.vectorize_inputs(sequences=sequences,
                                                             max_sequence_size=Configuration['sampling']['max_sequence_size'])

            targets = np.zeros((len(sequences), len(self.label_ids)), dtype=np.int32)
            for i, (document_tags) in enumerate(tags):
                for tag in document_tags:
                    if tag in self.label_ids:
                        targets[i][self.label_ids[tag]] = 1

        del sequences, tags

        if 'elmo' in Configuration['model']['token_encoding']:
            return [samples, samples2], targets

        return samples[:,:512,:],targets # bert&BERT, 

    def train(self,create_new_generator):
        LOGGER.info('\n---------------- Train Starting ----------------')

        for param_name, value in Configuration['model'].items():
            LOGGER.info('\t{}: {}'.format(param_name, value))

        # Load training/validation data
        LOGGER.info('Load training/validation data')
        LOGGER.info('------------------------------')
        model_type=Configuration['model']['architecture']# BERT ,to set max length

        train_val_generator_fn="data/generators/train_val_generator_{}_{}.pickle".format(model_type.lower(),Configuration['model']['batch_size'])
       
        if (os.path.exists(train_val_generator_fn)and (not create_new_generator)):
            print("train val dataloaders alreay exist, load them now.")
            with open(train_val_generator_fn, "rb") as f:
                train_generator,val_generator = pickle.load(f) 
        else:

            val_documents = self.load_dataset('dev',model_type)
            val_samples, val_tags = self.process_dataset(val_documents)
            val_generator = SampleGenerator(val_samples, val_tags, experiment=self,
                                            batch_size=Configuration['model']['batch_size'])
            # pdb.set_trace()
            documents = self.load_dataset('train',model_type)
            train_samples, train_tags = self.process_dataset(documents)
            train_generator = SampleGenerator(train_samples, train_tags, experiment=self,
                                            batch_size=Configuration['model']['batch_size'])
            
            # pdb.set_trace()
            with open(train_val_generator_fn, "wb") as f:
                pickle.dump((train_generator, val_generator),f)
            # pdb.set_trace()
        
        
        
        
        # strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():
        
        LOGGER.info('Compile neural network')
        LOGGER.info('------------------------------')
        if 'label' in Configuration['model']['architecture'].lower():
            network = LabelDrivenClassification(self.label_terms_ids)
        else:
            network = DocumentClassification(self.label_terms_ids)

        network.compile(n_hidden_layers=Configuration['model']['n_hidden_layers'],
                        hidden_units_size=Configuration['model']['hidden_units_size'],
                        dropout_rate=Configuration['model']['dropout_rate'],
                        word_dropout_rate=Configuration['model']['word_dropout_rate'],
                        lr=Configuration['model']['lr'])

        network.model.summary(line_length=200, print_fn=LOGGER.info)

        with tempfile.NamedTemporaryFile(delete=True) as w_fd:
            weights_file = w_fd.name

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(filepath=weights_file, monitor='val_loss', mode='auto',
                                           verbose=1, save_best_only=True, save_weights_only=True)
        batch_size=Configuration['model']['batch_size']
        
        # Fit model
        LOGGER.info('Fit model')
        LOGGER.info('-----------')
        start_time = time.time()
        # pdb.set_trace()

        # if model_type == "bert":
        #     print("truncate sequence for bert model at 500th token") 
        #     train_generator=Generator_proxy(train_generator)
        #     val_generator=Generator_proxy(val_generator)

        # else:
        #     raise Error

        val_samples_tag_fn="data/generators/val_samples_tag_{}_{}.pickle".format(model_type.lower(),Configuration['model']['batch_size'])
        if (os.path.exists(val_samples_tag_fn)and (not create_new_generator)):
            print("val samples and val tags alreay exist, load them now.")
            with open(val_samples_tag_fn, "rb") as f:
                val_documents,val_samples, val_targets = pickle.load(f) 
        else:
            val_documents = self.load_dataset('dev',model_type)# TODO rebundary, should be in the previous pickle 
            val_samples, val_tags = self.process_dataset(val_documents)# TODO rebundary, should be in the previous pickle 
            val_samples, val_targets = self.encode_dataset(val_samples, val_tags)
            with open(val_samples_tag_fn, "wb") as f:
                    pickle.dump((val_documents,val_samples, val_targets),f)


        try:
            fit_history = network.model.fit_generator(train_generator,
                                                    validation_data=val_generator,
                                                    workers=os.cpu_count()//2,
                                                    steps_per_epoch=len(train_generator),
                                                    epochs=Configuration['model']['epochs'],
                                                    callbacks=[early_stopping, model_checkpoint,Calculate_performance(val_samples, val_targets)],
                                                        verbose = True)
        except KeyboardInterrupt:
            LOGGER.info("skip rest of training")
            
        network.model.save(os.path.join(MODELS_DIR, '{}.h5'.format('{}_{}_{}'.format(
            Configuration['task']['dataset'].upper(), 'HIERARCHICAL' if Configuration['sampling']['hierarchical'] else 'FLAT',
            Configuration['model']['architecture'].upper()))))

        best_epoch = np.argmin(fit_history.history['val_loss']) + 1
        n_epochs = len(fit_history.history['val_loss'])
        val_loss_per_epoch = '- ' + ' '.join('-' if fit_history.history['val_loss'][i] < np.min(fit_history.history['val_loss'][:i])
                                             else '+' for i in range(1, len(fit_history.history['val_loss'])))
        LOGGER.info('\nBest epoch: {}/{}'.format(best_epoch, n_epochs))
        LOGGER.info('Val loss per epoch: {}\n'.format(val_loss_per_epoch))

        del train_generator

        LOGGER.info('Load valid data')
        LOGGER.info('------------------------------')

        
        # network.model=load_model(os.path.join(MODELS_DIR, '{}.h5'.format('{}_{}_{}'.format(
        #     Configuration['task']['dataset'].upper(), 'HIERARCHICAL' if Configuration['sampling']['hierarchical'] else 'FLAT',
        #     Configuration['model']['architecture'].upper()))), custom_objects={'BERT': BERT})




        self.calculate_performance(model=network.model, true_samples=val_samples, true_targets=val_targets)

        LOGGER.info('Load test data')
        LOGGER.info('------------------------------')

        test_documents = self.load_dataset('test',model_type)
        limit = len(test_documents) % Configuration['model']['batch_size'] if Configuration['model']['architecture'] == 'BERT' else 0
        test_samples, test_tags = self.process_dataset(test_documents if not limit else test_documents[:-limit])
        test_samples, test_targets = self.encode_dataset(test_samples, test_tags)

        self.calculate_performance(model=network.model, true_samples=test_samples, true_targets=test_targets)

        total_time = time.time() - start_time
        LOGGER.info('\nTotal Training Time: {} hours'.format(total_time/3600))

    def calculate_performance(self, model, true_samples, true_targets,verbose=True):

        predictions = model.predict(true_samples,
                                            batch_size=Configuration['model']['batch_size']
                                            if Configuration['model']['architecture'] == 'BERT'
                                               or Configuration['model']['token_encoding'] == 'elmo' else None)

        pred_targets = (predictions > 0.5).astype('int32')

        template = 'R@{} : {:1.3f}   P@{} : {:1.3f}   RP@{} : {:1.3f}   NDCG@{} : {:1.3f}'

        # Overall
        for labels_range, frequency, message in zip(self.margins,
                                                    ['Overall', 'Frequent', 'Few', 'Zero'],
                                                    ['Overall', 'Frequent Labels (>=50 Occurrences in train set)',
                                                     'Few-shot (<=50 Occurrences in train set)',
                                                     'Zero-shot (No Occurrences in train set)']):
            start, end = labels_range
            LOGGER.info(message)
            LOGGER.info('----------------------------------------------------')
            for average_type in ['micro', 'macro', 'weighted']:
                p = precision_score(true_targets[:, start:end], pred_targets[:, start:end], average=average_type)
                r = recall_score(true_targets[:, start:end], pred_targets[:, start:end], average=average_type)
                f1 = f1_score(true_targets[:, start:end], pred_targets[:, start:end], average=average_type)
                LOGGER.info('{:8} - Precision: {:1.4f}   Recall: {:1.4f}   F1: {:1.4f}'.format(average_type, p, r, f1))

            if verbose:
                for i in range(1, Configuration['sampling']['evaluation@k'] + 1):
                    r_k = mean_recall_k(true_targets[:, start:end], predictions[:, start:end], k=i)
                    p_k = mean_precision_k(true_targets[:, start:end], predictions[:, start:end], k=i)
                    rp_k = mean_rprecision_k(true_targets[:, start:end], predictions[:, start:end], k=i)
                    ndcg_k = mean_ndcg_score(true_targets[:, start:end], predictions[:, start:end], k=i)
                    LOGGER.info(template.format(i, r_k, i, p_k, i, rp_k, i, ndcg_k))
                LOGGER.info('----------------------------------------------------')


class Calculate_performance(Callback):
    """

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

    def __init__(self, true_samples, true_targets):
        super(Calculate_performance, self).__init__()
        self.true_samples = true_samples
        self.true_targets=true_targets

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.true_samples,
                                            batch_size=Configuration['model']['batch_size']
                                            if Configuration['model']['architecture'] == 'BERT'
                                               or Configuration['model']['token_encoding'] == 'elmo' else None)
        pred_targets = (predictions > 0.5).astype('int32')

        
        outfile = TemporaryFile()
        np.save(outfile, predictions)
        LOGGER.info("predictions is saved to{}".format(outfile))

        template = 'R@{} : {:1.3f}   P@{} : {:1.3f}   RP@{} : {:1.3f}   NDCG@{} : {:1.3f}'
        for average_type in ['micro', 'macro', 'weighted']:
            p = precision_score(self.true_targets, pred_targets, average=average_type)
            r = recall_score(self.true_targets, pred_targets, average=average_type)
            f1 = f1_score(self.true_targets, pred_targets, average=average_type)
            LOGGER.info('{:8} - Precision: {:1.4f}   Recall: {:1.4f}   F1: {:1.4f}'.format(average_type, p, r, f1))


class SampleGenerator(Sequence):
    '''
    Generates data for Keras
    :return: x_batch, y_batch
    '''

    def __init__(self, samples, targets, experiment, batch_size=32, shuffle=True):
        """Initialization"""
        self.data_samples = samples
        self.targets = targets # actually tags
        self.batch_size = batch_size
        self.indices = np.arange(len(samples))
        self.experiment = experiment
        self.shuffle = shuffle

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.data_samples) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of batch's sequences + tags
        samples = [self.data_samples[k] for k in indices]
        targets = [self.targets[k] for k in indices]
        # Vectorize inputs (x,y)
        x_batch, y_batch = self.experiment.encode_dataset(samples, targets)# targets are actually tags

        return x_batch, y_batch

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)

# class Generator_proxy(Sequence):

#     def __init__(self, generator):
#         self.underhood = generator

#     def __len__(self):
#         return self.underhood.__len__()

#     def __getitem__(self, index):
#         x_batch, y_batch=self.underhood.__getitem__(index)
#         x_batch=x_batch[:,:512,:]
#         return x_batch, y_batch

#     def on_epoch_end(self):
#         self.underhood.on_epoch_end()

if __name__ == '__main__':
    # configure
    # configuration_fn="data/configuration.pickle"
    # if os.path.exists(configuration_fn):
    #     print("configuration alreay initiated, load them now.")
    #     with open(configuration_fn, "rb") as f:
    #         Configuration = pickle.load(f) 
    # else:
    #     Configuration.configure()
    #     with open(configuration_fn, "wb") as f:
    #         pickle.dump(Configuration,f)

    # # new classifier
    # lmTextClassifier=LMTC(Configuration)
    # # train
    # lmTextClassifier.train()
    import argparse, sys

    parser = argparse.ArgumentParser()
    #首先是mandatory parameters
    parser.add_argument('--create_new_generator', action='store_true', help='create_new_generator')
    args = parser.parse_args()


    create_new_generator=args.create_new_generator
    Configuration.configure()
    LMTC().train(create_new_generator)
