import logging
import tensorflow_hub as hub
from tensorflow.keras.layers import Layer, Lambda
import tensorflow.keras.backend as K
from configuration import Configuration
from typing import List

try:
    from .bert_tokenization import FullTokenizer
except:
    print('if you want to use Google\'s encoder and pretrained models, please clone the bert submodule')


LOGGER = logging.getLogger(__name__)


class BERT(Layer):
    def __init__(self, output_representation=0, **kwargs):
        self.bert = None
        super(BERT, self).__init__(**kwargs)

        if output_representation:
            self.output_representation = 'sequence_output'
        else:
            self.output_representation = 'pooled_output'

    def build(self, input_shape):
        if Configuration['model']['bert'] == 'bertbase':
            self.bert = hub.Module('https://tfhub.dev/google/bert_{}_L-12_H-768_A-12/1'.format(Configuration['model']['bert_case']),
                                   trainable=True, name="{}_module".format(self.name))
        else:
            raise Exception('Unsupported bert module: "{}". Valid modules are: bertbase'.format(Configuration['model']['bert']))

        # Remove unused layers and set trainable parameters
        self.trainable_weights += [var for var in self.bert.variables
                                   if not "/cls/" in var.name and not "/pooler/" in var.name]
        super(BERT, self).build(input_shape)

    def call(self, x, mask=None):

        splits = Lambda(lambda k: K.tf.split(k, num_or_size_splits=3, axis=2))(x)

        inputs = []
        for i in range(len(splits)):
            inputs.append(Lambda(lambda s: K.tf.squeeze(s, axis=-1), name='squeeze_{}'.format(i))(splits[i]))

        outputs = self.bert(dict(input_ids=inputs[0], input_mask=inputs[1], segment_ids=inputs[2]),
                            as_dict=True, signature='tokens')[
            'sequence_output']

        if self.output_representation == 'pooled_output':
            return K.tf.squeeze(outputs[:, 0:1, :], axis=1)
        else:
            return outputs

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        if self.output_representation == 'pooled_output':
            return (None, 768)
        else:
            return (None, None, 768)


class TextEncoder:

    def __init__(self, vocab_size: int):
        # NOTE you MUST always put unk at 0, then regular vocab, then special tokens, and then pos
        self.vocab_size = vocab_size
        self.unk_id = 0

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, sent: str) -> List[int]:
        raise NotImplementedError()


class BERTTextEncoder(TextEncoder):
    def __init__(self, vocab_file: str, do_lower_case: bool = True, max_len=512) -> None:
        self.tokenizer = FullTokenizer(vocab_file, do_lower_case)
        super().__init__(len(self.tokenizer.vocab))
        self.max_len = max_len
        self.bert_unk_id = self.tokenizer.vocab['[UNK]']
        self.bert_msk_id = self.tokenizer.vocab['[MASK]']
        self.bert_cls_id = self.tokenizer.vocab['[CLS]']
        self.bert_sep_id = self.tokenizer.vocab['[SEP]']

    def encode(self, sent: str) -> List[int]:
        return [self.bert_cls_id] + \
               self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sent))[:self.max_len-2] + \
               [self.bert_sep_id]

