import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.datasets import imdb

data = imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words = 10000)

# A dictionary mapping words to an integer index
_word_index = imdb.get_word_index()

word_index = {k: (v+3) for k, v in _word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# trim data
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data, value = word_index["<PAD>"], padding = 'post', maxlen = 250
)
test_data = keras.preprocessing.sequence.pad_sequences(
    test_data, value = word_index["<PAD>"], padding = 'post', maxlen = 250
)


# this function will return the decoded (human readable) reviews
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


# building model
model = keras.Sequential()
model.add(keras.layers.Embedding(10000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation = 'relu'))
model.add(keras.layers.Dense(1, activation = 'sigmoid'))
