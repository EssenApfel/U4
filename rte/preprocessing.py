import json

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Vocab:

    def __init__(self):
        self.token_index = {}
        self.index_token = {}

    def fit(self, labels):
        self.token_index = {label: i for i, label in enumerate(set(labels))}
        self.index_token = {v: k for k, v in self.token_index.items()}
        return self

    def encode(self, labels):
        label_ids = [self.token_index.get(label) for label in labels]
        return label_ids

    def decode(self, label_ids):
        labels = [self.index_token.get(label_id) for label_id in label_ids]
        return labels

    @property
    def size(self):
        """Return vocabulary size."""
        return len(self.token_index)

    def save(self, file_path):
        with open(file_path, 'w') as f:
            config = {
                'token_index': self.token_index,
                'index_token': self.index_token
            }
            f.write(json.dumps(config))

    @classmethod
    def load(cls, file_path):
        with open(file_path) as f:
            config = json.load(f)
            vocab = cls()
            vocab.token_index = config.token_index
            vocab.index_token = config.index_token
        return vocab

def convert_examples_to_features(x, y,
                                 vocab,
                                 max_seq_length,
                                 tokenizer):
    features = {
        'input_ids': [],
        'attention_mask': [],
        'token_type_ids': [],
        'label_ids': np.asarray(vocab.encode(y))
    }
    for pairs in x:
        tokens = [tokenizer.cls_token]
        token_type_ids = []
        for i, sent in enumerate(pairs):
            word_tokens = tokenizer.tokenize(sent)
            tokens.extend(word_tokens)
            tokens += [tokenizer.sep_token]
            len_sent = len(word_tokens) + 1
            token_type_ids += [i] * len_sent

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        features['input_ids'].append(input_ids)
        features['attention_mask'].append(attention_mask)
        features['token_type_ids'].append(token_type_ids)

    for name in ['input_ids', 'attention_mask', 'token_type_ids']:
        features[name] = pad_sequences(features[name], padding='post', maxlen=max_seq_length)

    x = [features['input_ids'], features['attention_mask'], features['token_type_ids']]
    y = features['label_ids']
    return x, y