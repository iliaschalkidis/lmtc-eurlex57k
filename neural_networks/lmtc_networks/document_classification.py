import os

import numpy as np
from gensim.models import KeyedVectors
from keras import backend as K
from keras.layers import Bidirectional, GlobalMaxPooling1D, Dropout
from keras.layers import CuDNNGRU, GRU
from keras.layers import Dense, Embedding, add, concatenate
from keras.layers import Input, SpatialDropout1D
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.constraints import MinMaxNorm

from data import VECTORS_DIR
from configuration import Configuration
from neural_networks.layers import Camouflage, Attention, ContextualAttention, BERT
from neural_networks.layers import TimestepDropout, SymmetricMasking, ElmoEmbeddingLayer


class DocumentClassification:
    def __init__(self, label_terms_ids):
        super().__init__()
        self._cuDNN = Configuration['model']['cuDNN_acceleration']
        self._decision_type = Configuration['task']['decision_type']
        self.elmo = True if 'elmo' in Configuration['model']['token_encoding'] else False
        self.n_classes = len(label_terms_ids)
        self._attention_mechanism = Configuration['model']['attention_mechanism']
        if 'word2vec' in Configuration['model']['token_encoding']:
            self.word_embedding_path = os.path.join(VECTORS_DIR, Configuration['model']['embeddings'])

    def __del__(self):
        K.clear_session()
        del self.model

    def PretrainedEmbedding(self):

        inputs = Input(shape=(None,), dtype='int32')
        embeddings = KeyedVectors.load_word2vec_format(self.word_embedding_path, binary=False)
        word_encodings_weights = np.concatenate((np.zeros((1, embeddings.syn0.shape[-1]), dtype=np.float32), embeddings.syn0), axis=0)
        embeds = Embedding(len(word_encodings_weights), word_encodings_weights.shape[-1],
                           weights=[word_encodings_weights], trainable=False)(inputs)

        return Model(inputs=inputs, outputs=embeds, name='embedding')

    def compile(self, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate, lr):

        shape = (Configuration['sampling']['max_sequences_size'], Configuration['sampling']['max_sequence_size'])
        if Configuration['sampling']['hierarchical']:
            self._compile_hans(shape=shape, n_hidden_layers=n_hidden_layers, hidden_units_size=hidden_units_size,
                               dropout_rate=dropout_rate, word_dropout_rate=word_dropout_rate, lr=lr)
        elif Configuration['model']['architecture'] == 'BERT':
            self._compile_bert(shape=shape, dropout_rate=dropout_rate, lr=lr)
        elif Configuration['model']['attention_mechanism']:
            self._compile_bigrus_attention(shape=shape, n_hidden_layers=n_hidden_layers, hidden_units_size=hidden_units_size,
                                           dropout_rate=dropout_rate, word_dropout_rate=word_dropout_rate, lr=lr)
        else:
            self._compile_bigrus(n_hidden_layers=n_hidden_layers, hidden_units_size=hidden_units_size,
                                 dropout_rate=dropout_rate, word_dropout_rate=word_dropout_rate, lr=lr)

    def _compile_bigrus(self, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate, lr):
        """
        Compiles a Hierarchical RNN based on the given parameters
        :param hidden_units_size: size of hidden units, as a list
        :param dropout_rate: The percentage of inputs to dropout
        :param word_dropout_rate: The percentage of timesteps to dropout
        :param lr: learning rate
        :return: Nothing
        """

        # Document Feature Representation
        if self.elmo:
            document_inputs = Input(shape=(1, ), dtype='string', name='document_inputs')
            document_elmos = ElmoEmbeddingLayer()(document_inputs)
            document_inputs2 = Input(shape=(None,), name='document_inputs2')
            self.pretrained_embeddings = self.PretrainedEmbedding()
            document_embs = self.pretrained_embeddings(document_inputs2)
            doc_embs = concatenate([document_embs, document_elmos])

        else:
            document_inputs = Input(shape=(None,), name='document_inputs')
            self.pretrained_embeddings = self.PretrainedEmbedding()
            doc_embs = self.pretrained_embeddings(document_inputs)

        # Apply variational dropout
        drop_doc_embs = SpatialDropout1D(dropout_rate, name='feature_dropout')(doc_embs)
        encodings = TimestepDropout(word_dropout_rate, name='word_dropout')(drop_doc_embs)

        # Bi-GRUs over token embeddings
        return_sequences = True
        for i in range(n_hidden_layers):
            if i == n_hidden_layers - 1:
                return_sequences = False
            if self._cuDNN:
                grus = Bidirectional(CuDNNGRU(hidden_units_size, return_sequences=return_sequences), name='bidirectional_grus_{}'.format(i))(encodings)
            else:
                grus = Bidirectional(GRU(hidden_units_size, activation="tanh", recurrent_activation='sigmoid',
                                         return_sequences=return_sequences), name='bidirectional_grus_{}'.format(i))(encodings)
            if i != n_hidden_layers - 1:
                grus = Camouflage(mask_value=0.0)([grus, encodings])
                if i == 0:
                    encodings = SpatialDropout1D(dropout_rate)(grus)
                else:
                    encodings = add([grus, encodings])
                    encodings = SpatialDropout1D(dropout_rate)(encodings)
            else:
                encodings = grus

        # Final output (projection) layer
        outputs = Dense(self.n_classes, activation='sigmoid' if self._decision_type == 'multi_label' else 'softmax',
                        name='outputs')(encodings)

        # Wrap up model + Compile with optimizer and loss function
        self.model = Model(inputs=document_inputs if not self.elmo else [document_inputs, document_inputs2],
                           outputs=[outputs])
        self.model.compile(optimizer=Adam(lr=lr, clipvalue=5.0),
                           loss='binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy')

    def _compile_bigrus_attention(self, shape, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate, lr):
        """
        Compiles a Hierarchical RNN based on the given parameters
        :param hidden_units_size: size of hidden units, as a list
        :param dropout_rate: The percentage of inputs to dropout
        :param word_dropout_rate: The percentage of timesteps to dropout
        :param lr: learning rate
        :return: Nothing
        """

        # Document Feature Representation
        if self.elmo:
            document_inputs = Input(shape=(1, ), dtype='string', name='document_inputs')
            document_elmos = ElmoEmbeddingLayer()(document_inputs)
            document_inputs2 = Input(shape=(None,), name='document_inputs2')
            self.pretrained_embeddings = self.PretrainedEmbedding()
            document_embs = self.pretrained_embeddings(document_inputs2)
            doc_embs = concatenate([document_embs, document_elmos])

        else:
            document_inputs = Input(shape=(None,), name='document_inputs')
            self.pretrained_embeddings = self.PretrainedEmbedding()
            doc_embs = self.pretrained_embeddings(document_inputs)

        # Apply variational dropout
        drop_doc_embs = SpatialDropout1D(dropout_rate, name='feature_dropout')(doc_embs)
        encodings = TimestepDropout(word_dropout_rate, name='word_dropout')(drop_doc_embs)

        # Bi-GRUs over token embeddings
        for i in range(n_hidden_layers):
            if self._cuDNN:
                grus = Bidirectional(CuDNNGRU(hidden_units_size, return_sequences=True), name='bidirectional_grus_{}'.format(i))(encodings)
            else:
                grus = Bidirectional(GRU(hidden_units_size, activation="tanh", recurrent_activation='sigmoid',
                                         return_sequences=True), name='bidirectional_grus_{}'.format(i))(encodings)
            grus = Camouflage(mask_value=0.0)([grus, encodings])
            if i == 0:
                encodings = SpatialDropout1D(dropout_rate)(grus)
            else:
                encodings = add([grus, encodings])
                encodings = SpatialDropout1D(dropout_rate)(encodings)

        # Attention over BI-GRU (context-aware) embeddings
        if self._attention_mechanism == 'maxpooling':
            doc_encoding = GlobalMaxPooling1D(name='max_pooling')(encodings)
        elif self._attention_mechanism == 'attention':
            # Mask encodings before attention
            grus_outputs = SymmetricMasking(mask_value=0, name='masking')([encodings, encodings])
            doc_encoding = Attention(kernel_regularizer=l2(), bias_regularizer=l2(), name='self_attention')(grus_outputs)
        losses = 'binary_crossentropy' \
            if self._decision_type == 'multi_label' else 'categorical_crossentropy'
        loss_weights = None

        # Final output (projection) layer
        outputs = Dense(self.n_classes, activation='sigmoid' if self._decision_type == 'multi_label' else 'softmax', name='outputs')(doc_encoding)

        # Wrap up model + Compile with optimizer and loss function
        self.model = Model(inputs=document_inputs if not self.elmo else [document_inputs, document_inputs2],
                           outputs=[outputs])
        self.model.compile(optimizer=Adam(lr=lr, clipvalue=5.0), loss=losses, loss_weights=loss_weights)

    def _compile_hans(self, shape, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate, lr):
        """
        Compiles a Hierarchical Attention Network based on the given parameters
        :param shape: The shape of the sequence, i.e. (number of sections, number of tokens)
        :param hidden_units_size: size of hidden units, as a list
        :param dropout_rate: The percentage of inputs to dropout
        :param word_dropout_rate: The percentage of timesteps to dropout
        :param lr: learning rate
        :return: Nothing
        """

        # Sentence Feature Representation
        section_inputs = Input(shape=(None,), name='document_inputs')
        self.pretrained_embeddings = self.PretrainedEmbedding()
        section_embs = self.pretrained_embeddings(section_inputs)

        # Apply variational dropout
        drop_section_embs = SpatialDropout1D(dropout_rate, name='feature_dropout')(section_embs)
        encodings = TimestepDropout(word_dropout_rate, name='word_dropout')(drop_section_embs)

        # Bi-GRUs over token embeddings
        for i in range(n_hidden_layers[0]):
            if self._cuDNN:
                grus = Bidirectional(CuDNNGRU(hidden_units_size[0], return_sequences=True, kernel_constraint=MinMaxNorm(min_value=-2, max_value=2)),
                                     name='bidirectional_grus_{}'.format(i))(encodings)
            else:
                grus = Bidirectional(GRU(hidden_units_size[0], activation="tanh", recurrent_activation='sigmoid',
                                         return_sequences=True), kernel_constraint=MinMaxNorm(min_value=-2, max_value=2),
                                     name='bidirectional_grus_{}'.format(i))(encodings)
            grus = Camouflage(mask_value=0.0)([grus, encodings])
            if i == 0:
                encodings = SpatialDropout1D(dropout_rate)(grus)
            else:
                encodings = add([grus, encodings])
                encodings = SpatialDropout1D(dropout_rate)(encodings)

        # Attention over BI-GRU (context-aware) embeddings
        if self._attention_mechanism == 'maxpooling':
            section_encoder = GlobalMaxPooling1D()(encodings)
        elif self._attention_mechanism == 'attention':
            encodings = SymmetricMasking()([encodings, encodings])
            section_encoder = ContextualAttention(kernel_regularizer=l2(), bias_regularizer=l2())(encodings)

        # Wrap up section_encoder
        section_encoder = Model(inputs=section_inputs, outputs=section_encoder, name='sentence_encoder')

        # Document Input Layer
        document_inputs = Input(shape=(shape[0], shape[1],), name='document_inputs')

        # Distribute sentences
        section_encodings = TimeDistributed(section_encoder, name='sentence_encodings')(document_inputs)

        # BI-GRUs over section embeddings
        for i in range(n_hidden_layers[1]):
            if self._cuDNN:
                grus = Bidirectional(CuDNNGRU(hidden_units_size[1], return_sequences=True,
                                              kernel_constraint=MinMaxNorm(min_value=-2, max_value=2)),
                                     name='bidirectional_grus_upper_{}'.format(i))(section_encodings)
            else:
                grus = Bidirectional(GRU(hidden_units_size[1], activation="tanh", recurrent_activation='sigmoid',
                                         return_sequences=True, kernel_constraint=MinMaxNorm(min_value=-2, max_value=2)),
                                     name='bidirectional_grus_upper_{}'.format(i))(section_encodings)
            grus = Camouflage(mask_value=0.0)([grus, section_encodings])
            if i == 0:
                section_encodings = SpatialDropout1D(dropout_rate)(grus)
            else:
                section_encodings = add([grus, section_encodings])
                section_encodings = SpatialDropout1D(dropout_rate)(section_encodings)

        # Attention over BI-LSTM (context-aware) sentence embeddings
        if self._attention_mechanism == 'maxpooling':
            doc_encoding = GlobalMaxPooling1D(name='max_pooling')(section_encodings)
        elif self._attention_mechanism == 'attention':
            section_encodings = SymmetricMasking()([section_encodings, section_encodings])
            doc_encoding = ContextualAttention(kernel_regularizer=l2(), bias_regularizer=l2(), name='self_attention')(
                section_encodings)
        losses = 'binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy'
        loss_weights = None

        # Final output (projection) layer
        outputs = Dense(self.n_classes, activation='sigmoid' if self._decision_type == 'multi_label' else 'softmax', name='outputs')(doc_encoding)

        # Wrap up model + Compile with optimizer and loss function
        self.model = Model(inputs=document_inputs,
                           outputs=[outputs])
        self.model.compile(optimizer=Adam(lr=lr, clipvalue=2.0), loss=losses, loss_weights=loss_weights)

    def _compile_bert(self, shape, dropout_rate, lr):

        word_inputs = Input(shape=(None, 3), dtype='int32')
        doc_encoding = BERT()(word_inputs)

        doc_encoding = Dropout(dropout_rate)(doc_encoding)

        # Final output (projection) layer
        outputs = Dense(self.n_classes, activation='sigmoid' if self._decision_type == 'multi_label' else 'softmax',
                        name='outputs')(doc_encoding)

        losses = 'binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy'
        loss_weights = None

        # Wrap up model + Compile with optimizer and loss function
        self.model = Model(inputs=word_inputs, outputs=[outputs])
        self.model.compile(optimizer=Adam(lr=lr), loss=losses, loss_weights=loss_weights)

