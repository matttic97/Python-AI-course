import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.datasets import imdb

data = imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words = 88000)

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


# this function will return the encoded reviews
def encode_review(text):
    encoded = [1]

    for word in text:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)

    return encoded


'''# building model
model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation = 'relu'))
model.add(keras.layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs = 40, batch_size = 512, validation_data = (x_val, y_val), verbose = 1)

results = model.evaluate(test_data, test_labels)
print(results)

# save model
model.save('model.h5')'''

# load model
model = keras.models.load_model('model.h5')

with open('data/text.txt', encoding = 'utf-8') as f:
    for line in f.readlines():
        nline = line.replace(',', '').replace('.', '').replace('(', '')\
            .replace(')', '').replace(':', '').replace('"', '').strip().split(' ')
        encode = encode_review(nline)
        encode = keras.preprocessing.sequence.pad_sequences(
            [encode], value = word_index["<PAD>"], padding = 'post', maxlen = 250
        )
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])


test_review = test_data[0]
predict = model.predict([test_review])
print('Review:')
print(decode_review(test_review))
print('Prediction:', str(predict[0]))
print('Actual:', str(test_labels[0]))
