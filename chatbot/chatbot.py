import nltk


from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow as tf
import random
import json
import pickle

import os
os.chdir("/Users/farihatanjin/")

from flask import Flask
app = Flask(__name__)
from app import routes



with open("package.json") as file:
    data = json.load(file)

try:
    with open("data.pickle","rb") as f:
        words,labels,training,output = pickle.load(f)
except:

    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern) #store all words in a list
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    #stemmer.stem removes morphological affixes so we are generating a unique list of stemmed words here
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    #one hot encoding

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x,doc in enumerate(docs_x):
        bag = []

        stem_wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in stem_wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tf.compat.v1.reset_default_graph()

#neural network
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

#train and save model
if os.path.exists("/Users/farihatanjin/model.tflearn.meta"):
    model.load("model.tflearn")
else:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("mode.tflearn")

#function to prepare input data
def bag_of_words(s,words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for word in s_words:
        for i,w in enumerate(words):
            if w == word:
                bag[i] = 1

    return numpy.array(bag)

#function to take in user input. outputs an answer from tag with the highest probability

@app.route('/')
@app.route('/userInput()')
def userInput():
    print("Chat with me!")
    while True:
        info = input("You: ")
        if info.lower() == "quit":
            break
        results = model.predict([bag_of_words(info,words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7:
            for tag_val in data["intents"]:
              if tag_val['tag'] == tag:
                  responses = tag_val["responses"]

            print(random.choice(responses))
        else:
            print("I didn't get that, could you try again?")

