import tensorflow as tf
from transformers import BertConfig, TFBertForSequenceClassification, BertJapaneseTokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from preprocessing import convert_examples_to_features, Vocab
from utils import load_dataset, evaluate

def build_model(pretrained_model_name_or_path, num_labels):
    config = BertConfig.from_pretrained(
        pretrained_model_name_or_path,
        num_labels=num_labels
    )
    model = TFBertForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path,
        config=config
    )
    model.layers[-1].activation = tf.keras.activations.softmax
    return model

def main():
    batch_size = 32
    epochs = 100
    model_path = 'models/'
    pretrained_model_name_or_path = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    maxlen = 100

    # Data loading.
    x, y = load_dataset('jrte-corpus/data/rte.lrec2020_sem_long.tsv')
    tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_model_name_or_path)

    # Pre-processing.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    target_vocab = Vocab().fit(y_train)
    features_train, labels_train = convert_examples_to_features(
        x_train,
        y_train,
        target_vocab,
        max_seq_length=maxlen,
        tokenizer=tokenizer
    )
    features_test, labels_test = convert_examples_to_features(
        x_test,
        y_test,
        target_vocab,
        max_seq_length=maxlen,
        tokenizer=tokenizer
    )

    # Dataset preparation for TensorFlow
    train_data = tf.data.Dataset.from_tensor_slices((features_train, labels_train)).batch(batch_size)
    test_data = tf.data.Dataset.from_tensor_slices((features_test, labels_test)).batch(batch_size)

    # Build model.
    model = build_model(pretrained_model_name_or_path, target_vocab.size)
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')

    # Callbacks
    callbacks = [
        EarlyStopping(patience=3),
    ]

    # Train the model.
    model.fit(train_data, epochs=epochs, validation_data=test_data, callbacks=callbacks)
    model.save_pretrained(model_path)

if __name__ == '__main__':
    main()