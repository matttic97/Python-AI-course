import os
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import random
import tensorflow as tf
import json
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
stemmer = LancasterStemmer()

with open('intents.json') as file:
    data = json.load(file)

try:
    with open('data.pickle', 'rb') as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            uniques = nltk.word_tokenize(pattern)
            words.extend(uniques)
            docs_x.append(uniques)
            docs_y.append(intent['tag'])

            if intent['tag'] not in labels:
                labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w != '?']
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        stems = [stemmer.stem(w) for w in doc]

        for word in words:
            if word in stems:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open('data.pickle', 'wb') as f:
        pickle.dump((words, labels, training, output), f)


tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]))
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load('model.tflearn')
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)

    model.save('model.tflearn')


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(w.lower()) for w in s_words if w != '?']

    for sw in s_words:
        for i, w in enumerate(words):
            if w == sw:
                bag[i] = 1

        return np.array(bag)


def chat():
    print('Say something (to quit type q):')
    while True:
        inp = input('You: ')
        if inp.lower() == 'q':
            break

        prediction = model.predict([bag_of_words(inp, words)])[0]
        l_index = np.argmax(prediction)

        if prediction[l_index] < 0.6:
            print("Didn't got this right, can you ask again?")
        else:
            tag = labels[l_index]
            for t in data['intents']:
                if t['tag'] == tag:
                    print(random.choice(t['responses']))


chat()
