import pandas as pd


def load_dataset(filepath, encoding='utf-8'):
    df = pd.read_csv(filepath,
                     encoding=encoding,
                     delimiter='\t',
                     names=['id', 'label', 't1', 't2', 'a', 'i'])

    return list(zip(df.t1, df.t2)), df.label


import numpy as np
from sklearn.metrics import classification_report


def evaluate(model, target_vocab, features, labels):
    label_ids = model.predict(features)
    label_ids = np.argmax(label_ids, axis=-1)
    y_pred = target_vocab.decode(label_ids)
    y_true = target_vocab.decode(labels)
    print(classification_report(y_true, y_pred, digits=4))
