from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class Attention(Layer):
    def __init__(self,
                 kernel_regularizer=None, bias_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False,
                 **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:

        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.


        Note: The layer has been tested with Keras 1.x

        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...

        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(kernel_regularizer)
        self.b_regularizer = regularizers.get(bias_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.return_attention = return_attention
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name))

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zeros',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.built = True

    def compute_mask(self, inputs, mask=None):
        # do not pass the mask to the next layers
        if self.return_attention:
            return [None, None]
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]


class ContextualAttention(Layer):
    def __init__(self,
                 kernel_regularizer=None, u_regularizer=None, bias_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False,
                 **kwargs):
        """
        Keras Layer that implements a context-aware Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:

        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.


        Note: The layer has been tested with Keras 1.x

        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...

        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(kernel_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(bias_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.return_attention = return_attention
        super(ContextualAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zeros',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(ContextualAttention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # do not pass the mask to the next layers
        if self.return_attention:
            return [None, None]
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)

        # Dot product with context vector U
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]


class LabelDrivenAttention(Layer):

    def __init__(self, kernel_regularizer=None, bias_regularizer=None, return_attention=False,
                 **kwargs):

        self.W_regularizer = regularizers.get(kernel_regularizer)
        self.b_regularizer = regularizers.get(bias_regularizer)
        self.init = initializers.get('he_normal')
        self.supports_masking = True
        self.return_attention = return_attention
        super(LabelDrivenAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape[0]) == 3
        assert len(input_shape[1]) == 2
        assert input_shape[0][-1] == input_shape[1][-1]

        self.W_d = self.add_weight(shape=(input_shape[1][-1], input_shape[0][-1]),
                                   initializer=self.init,
                                   regularizer=self.W_regularizer,
                                   name='{}_Wd'.format(self.name))

        self.b_d = self.add_weight(shape=(input_shape[1][-1],),
                                   initializer='zeros',
                                   regularizer=self.b_regularizer,
                                   name='{}_bd'.format(self.name))

        super(LabelDrivenAttention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # do not pass the mask to the next layers
        if self.return_attention:
            return [None, None]
        return None

    def call(self, x, mask=None):
        # Unfold inputs (document representations, label representations)
        doc_reps, label_reps = x

        doc2_reps = K.tanh(dot_product(doc_reps, self.W_d) + self.b_d)

        # Compute Attention Scores
        doc_a = dot_product(doc2_reps, label_reps)

        def label_wise_attention(values):
            doc_repi, ai = values
            ai = K.softmax(K.transpose(ai))
            label_aware_doc_rep = K.dot(ai, doc_repi)
            if self.return_attention:
                return [label_aware_doc_rep, ai]
            else:
                return [label_aware_doc_rep, label_aware_doc_rep]

        label_aware_doc_reprs, attention_scores = K.map_fn(label_wise_attention, [doc_reps, doc_a])

        label_aware_doc_reprs = K.sum(label_aware_doc_reprs * label_reps, axis=-1)
        label_aware_doc_reprs = K.sigmoid(label_aware_doc_reprs)

        if self.return_attention:
            return [label_aware_doc_reprs, attention_scores]

        return label_aware_doc_reprs

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0][0], input_shape[1][0]),
                    (input_shape[0][0], input_shape[1][0], input_shape[0][1])]
        return input_shape[0][0], input_shape[1][0]


class LabelWiseAttention(Layer):

    def __init__(self, kernel_regularizer=None, bias_regularizer=None,
                 return_attention=False, n_classes=4271, **kwargs):

        self.W_regularizer = regularizers.get(kernel_regularizer)
        self.b_regularizer = regularizers.get(bias_regularizer)
        self.init = initializers.get('he_normal')
        self.supports_masking = True
        self.return_attention = return_attention
        self.n_classes = n_classes
        super(LabelWiseAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.Wa = self.add_weight(shape=(self.n_classes, input_shape[-1]),
                                  initializer=self.init,
                                  regularizer=self.W_regularizer,
                                  name='{}_Wa'.format(self.name))

        self.Wo = self.add_weight(shape=(self.n_classes, input_shape[-1]),
                                  initializer=self.init,
                                  regularizer=self.W_regularizer,
                                  name='{}_Wo'.format(self.name))

        self.bo = self.add_weight(shape=(self.n_classes,),
                                  initializer='zeros',
                                  regularizer=self.b_regularizer,
                                  name='{}_bo'.format(self.name))

        super(LabelWiseAttention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # do not pass the mask to the next layers
        if self.return_attention:
            return [None, None]
        return None

    def call(self, x, mask=None):

        a = dot_product(x, self.Wa)

        def label_wise_attention(values):
            doc_repi, ai = values
            ai = K.softmax(K.transpose(ai))
            label_aware_doc_rep = K.dot(ai, doc_repi)
            if self.return_attention:
                return [label_aware_doc_rep, ai]
            else:
                return [label_aware_doc_rep, label_aware_doc_rep]

        label_aware_doc_reprs, attention_scores = K.map_fn(label_wise_attention, [x, a])

        # Compute label-scores
        label_aware_doc_reprs = K.sum(label_aware_doc_reprs * self.Wo, axis=-1) + self.bo
        label_aware_doc_reprs = K.sigmoid(label_aware_doc_reprs)

        if self.return_attention:
            return [label_aware_doc_reprs, attention_scores]

        return label_aware_doc_reprs

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], self.n_classes),
                    (input_shape[0], input_shape[1], self.n_classes, input_shape[-1])]
        return input_shape[0], self.n_classes
