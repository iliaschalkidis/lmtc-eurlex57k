from typing import List
import os
import re
import numpy as np

from data import VECTORS_DIR, DATA_DIR
from configuration import Configuration
from neural_networks.layers.bert import BERTTextEncoder


class Vectorizer(object):

    def __init__(self):
        pass

    def vectorize_inputs(self, sequences: List[List[str]], max_sequence_size=100, **kwargs):
        raise NotImplementedError


class BERTVectorizer(Vectorizer):

    def __init__(self):
        super().__init__()

    def vectorize_inputs(self, sequences: List[List[str]], max_sequence_size=100, **kwargs):

        bert_tokenizer = BERTTextEncoder(vocab_file=os.path.join(DATA_DIR, 'bert',
                                                                 Configuration['model']['bert'],
                                                                 'vocab.txt'),
                                         do_lower_case=True,
                                         max_len=max_sequence_size)

        token_indices = np.zeros((len(sequences), max_sequence_size), dtype=np.int32)
        seg_indices = np.zeros((len(sequences), max_sequence_size), dtype=np.int32)
        mask_indices = np.zeros((len(sequences), max_sequence_size), dtype=np.int32)

        for i, tokens in enumerate(sequences):
            bpes = bert_tokenizer.encode(' '.join([token for token in tokens]))
            limit = min(max_sequence_size, len(bpes))
            token_indices[i][:limit] = bpes[:limit]
            mask_indices[i][:limit] = np.ones((limit,), dtype=np.int32)

        return np.concatenate((np.reshape(token_indices, [len(sequences), max_sequence_size, 1]),
                               np.reshape(mask_indices, [len(sequences), max_sequence_size, 1]),
                               np.reshape(seg_indices, [len(sequences), max_sequence_size, 1])), axis=-1)


class ELMoVectorizer(Vectorizer):

    def __init__(self):
        super().__init__()

    def vectorize_inputs(self, sequences: List[List[str]], max_sequence_size=100, **kwargs):

        word_inputs = []
        # Encode ELMo embeddings
        for i, tokens in enumerate(sequences):
            sequence = ' '.join([token for token in tokens[:max_sequence_size]])
            if len(tokens) < max_sequence_size:
                sequence = sequence + ' ' + ' '.join(['#' for i in range(max_sequence_size - len(tokens))])
            word_inputs.append([sequence])
        return np.asarray(word_inputs)


class W2VVectorizer(Vectorizer):

    def __init__(self, w2v_model='glove.6B.200d.txt'):
        super().__init__()
        self.w2v_model = w2v_model

        self.word_indices = {'PAD': 0}
        count = 1
        with open(os.path.join(VECTORS_DIR, w2v_model)) as file:
            for line in file.readlines()[1:]:
                self.word_indices[line.split()[0]] = count
                count +=1

    def norm(self, token):
        """
        :return: normalized form of token text
        """
        if 'law2vec' in self.w2v_model:
            if token == '\n':
                return 'NL'
            return re.sub('\d', 'D', token.lower().strip(' '))
        else:
            return token.lower()

    def vectorize_inputs(self, sequences: List[List[str]], max_sequence_size=100, **kwargs):
        """
        Produce W2V indices for each token in the list of tokens
        :param sequences: list of lists of tokens
        :param max_sequence_size: maximum padding
        :param features: features to be considered
        """

        word_inputs = np.zeros((len(sequences), max_sequence_size, ), dtype=np.int32)

        for i, sentence in enumerate(sequences):
            for j, token in enumerate(sentence[:max_sequence_size]):
                if self.norm(token) in self.word_indices:
                    word_inputs[i][j] = self.word_indices[self.norm(token)]
                else:
                    word_inputs[i][j] = self.word_indices['unknown']
        
        return word_inputs
