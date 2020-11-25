import numpy as np
import os
from gensim.models import KeyedVectors
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Conv1D, Activation, Embedding
from tensorflow.keras.layers import Input, SpatialDropout1D
from tensorflow.keras.layers import GRU, add, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from data import VECTORS_DIR
from configuration import Configuration
from neural_networks.layers import Camouflage, GlobalMeanPooling1D, TimestepDropout, \
    SymmetricMasking, ElmoEmbeddingLayer, LabelDrivenAttention, LabelWiseAttention


class LabelDrivenClassification:
    def __init__(self, label_terms_ids):
        super().__init__()
        self._conf = 'NONE'
        self._cuDNN = Configuration['model']['cuDNN_acceleration']
        self._decision_type = Configuration['task']['decision_type']
        self.elmo = True if 'elmo' in Configuration['model']['token_encoding'] else False
        self.token_encoder = Configuration['model']['document_encoder']
        self.word_embedding_path = os.path.join(VECTORS_DIR, Configuration['model']['embeddings'])
        self.label_terms_ids = label_terms_ids

    def __del__(self):
        K.clear_session()
        del self.model

    def compile(self, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate, lr):

        if 'zero' in Configuration['model']['architecture']:
            self._compile_label_wise_attention_zero(n_hidden_layers=n_hidden_layers, hidden_units_size=hidden_units_size,
                                                    dropout_rate=dropout_rate,
                                                    word_dropout_rate=word_dropout_rate, lr=lr)
        else:
            self._compile_label_wise_attention(n_hidden_layers=n_hidden_layers, hidden_units_size=hidden_units_size,
                                               dropout_rate=dropout_rate,
                                               word_dropout_rate=word_dropout_rate, lr=lr)

    def PretrainedEmbedding(self):

        inputs = Input(shape=(None,), dtype='int32')
        embeddings = KeyedVectors.load_word2vec_format(self.word_embedding_path, binary=False)
        word_embeddings_weights = K.cast_to_floatx(np.concatenate((np.zeros((1, embeddings.syn0.shape[-1]), dtype=np.float32), embeddings.syn0), axis=0))
        embeds = Embedding(len(word_embeddings_weights), word_embeddings_weights.shape[-1],
                           weights=[word_embeddings_weights], trainable=False)(inputs)

        return Model(inputs=inputs, outputs=embeds, name='embedding')

    def TokenEncoder(self, inputs, encoder, dropout_rate, word_dropout_rate, hidden_layers, hidden_units_size):

        # Apply variational drop-out
        inner_inputs = SpatialDropout1D(dropout_rate)(inputs)
        inner_inputs = TimestepDropout(word_dropout_rate)(inner_inputs)

        if encoder == 'grus':
            # Bi-GRUs over token embeddings
            for i in range(hidden_layers):
                if self._cuDNN:
                    bi_grus = Bidirectional(CuDNNGRU(units=hidden_units_size, return_sequences=True))(inner_inputs)
                else:
                    bi_grus = Bidirectional(GRU(units=hidden_units_size, return_sequences=True, activation="tanh",
                                                recurrent_activation='sigmoid'))(inner_inputs)
                bi_grus = Camouflage(mask_value=0)(inputs=[bi_grus, inputs])
                if i == 0:
                    inner_inputs = SpatialDropout1D(dropout_rate)(bi_grus)
                else:
                    inner_inputs = add([bi_grus, inner_inputs])
                    inner_inputs = SpatialDropout1D(dropout_rate)(inner_inputs)

            outputs = Camouflage()([inner_inputs, inputs])
        elif encoder == 'cnns':
            # CNNs over token embeddings
            convs = Conv1D(filters=hidden_units_size, kernel_size=3, strides=1, padding="same")(inner_inputs)
            convs = Activation('tanh')(convs)
            convs = SpatialDropout1D(dropout_rate)(convs)
            outputs = Camouflage(mask_value=0)(inputs=[convs, inputs])

        return outputs

    def _compile_label_wise_attention(self, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate, lr):

        # Document Encoding
        if self.elmo:
            document_inputs = Input(shape=(1, ), dtype='string', name='document_inputs')
            document_elmos = ElmoEmbeddingLayer()(document_inputs)
            document_inputs2 = Input(shape=(None,), name='document_inputs2')
            self.pretrained_embeddings = self.PretrainedEmbedding()
            document_embs = self.pretrained_embeddings(document_inputs2)
            document_embs = concatenate([document_embs, document_elmos])

        else:
            document_inputs = Input(shape=(None,), name='document_inputs')
            self.pretrained_embeddings = self.PretrainedEmbedding()
            document_embs = self.pretrained_embeddings(document_inputs)
            
        document_ngram_encodings = self.TokenEncoder(inputs=document_embs, encoder=self.token_encoder, dropout_rate=dropout_rate,
                                                     word_dropout_rate=word_dropout_rate, hidden_layers=n_hidden_layers,
                                                     hidden_units_size=hidden_units_size)

        # Label-wise Attention Mechanism matching documents with labels
        document_label_encodings = LabelWiseAttention(n_classes=len(self.label_terms_ids))(document_ngram_encodings)
        losses = 'binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy'
        loss_weights = None

        self.model = Model(inputs=[document_inputs] if not self.elmo else [document_inputs, document_inputs2],
                            outputs=[document_label_encodings])
        self.model.compile(optimizer=Adam(lr=lr, clipvalue=5.0), loss=losses, loss_weights=loss_weights)

    def _compile_label_wise_attention_zero(self, n_hidden_layers, hidden_units_size, dropout_rate, word_dropout_rate, lr):

        # Document Encoding
        if self.elmo:
            document_inputs = Input(shape=(1, ), dtype='string', name='document_inputs')
            document_elmos = ElmoEmbeddingLayer()(document_inputs)
            document_inputs2 = Input(shape=(None,), name='document_inputs2')
            self.pretrained_embeddings = self.PretrainedEmbedding()
            document_embs = self.pretrained_embeddings(document_inputs2)
            document_embs = concatenate([document_embs, document_elmos])

        else:
            document_inputs = Input(shape=(None,), name='document_inputs')
            self.pretrained_embeddings = self.PretrainedEmbedding()
            document_embs = self.pretrained_embeddings(document_inputs)

        document_ngram_encodings = self.TokenEncoder(inputs=document_embs, encoder=self.token_encoder, dropout_rate=dropout_rate,
                                                     word_dropout_rate=word_dropout_rate, hidden_layers=n_hidden_layers,
                                                     hidden_units_size=hidden_units_size)

        # Labels Encoding
        if self.elmo:
            labels_inputs = Input(tensor=K.tf.convert_to_tensor(self.label_terms_ids, dtype='string'), name='label_inputs')
            labels_embs = ElmoEmbeddingLayer()(labels_inputs)
        else:
            labels_inputs = Input(tensor=K.tf.convert_to_tensor(self.label_terms_ids, dtype=K.tf.int32), name='label_inputs')
            labels_embs = self.pretrained_embeddings(labels_inputs)

        label_encodings = SpatialDropout1D(dropout_rate)(labels_embs)
        label_encodings = SymmetricMasking(mask_value=0.0)([label_encodings, label_encodings])
        label_encodings = GlobalMeanPooling1D()(label_encodings)

        # Label-wise Attention Mechanism matching documents with labels
        outputs = LabelDrivenAttention()([document_ngram_encodings, label_encodings])
        losses = 'binary_crossentropy' if self._decision_type == 'multi_label' else 'categorical_crossentropy'
        loss_weights = None

        # Compile network
        self.model = Model(inputs=[document_inputs, labels_inputs] if not self.elmo
                            else [document_inputs, document_inputs2, labels_inputs],
                            outputs=[outputs])
        self.model.compile(optimizer=Adam(lr=lr, clipvalue=5.0), loss=losses, loss_weights=loss_weights)
