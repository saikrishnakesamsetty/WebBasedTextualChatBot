import nltk
import speech_recognition as sr
from playsound import playsound
import os
from os import path
from gtts import gTTS
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import random
import json
import pickle


r = sr.Recognizer()
if os.path.exists('audio.mp3'):
    os.remove("audio.mp3")
i = 1
audio = 'null'
txt = 'null'

stemmer = LancasterStemmer()

with open("data.json") as file:
    data = json.load(file)

if(path.exists("data.pickle")):
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
else:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
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


net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
if(path.exists("model.tflearn.data-00000-of-00001") and path.exists("model.tflearn.index") and path.exists("model.tflearn.meta")):
    model.load("model.tflearn")
else:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chatbot(txt):
    #txt = 'null'
    #print("Start a conversation with Bot (say stop or good bye to stop)!")
    no_punct = ''
    Flag = True
    '''tts = gTTS("Hello! My name is Bot. I'm at your service!")
    tts.save('audio.mp3')
    print("Bot: Hello! My name is Bot... I'm at your service! :)")
    print()'''

    #playsound('audio.mp3')
    #os.remove("audio.mp3")
    tag = "unknown"
    '''with sr.Microphone() as source:
        print("Say Something")
        playsound('google_now_voice.mp3', True)
        audio = r.listen(source)
        print("time out, Thanks")
        playsound('google_glass_done.mp3', True)

        try:
            txt = r.recognize_google(audio)
            print("You: " + txt)'''
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    my_str = txt
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct = no_punct + char
    no_punct = no_punct.lower()

    #except:
        #pass;
    inp = no_punct

    results = model.predict([bag_of_words(inp, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    if results[0][results_index]<0.85:
        tag="unknown"
        fi = open("update.txt", "a+")
        fi.write("%s\n" % txt)
        fi.close()

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']


    res = random.choice(responses)
    '''tts = gTTS(res)
    tts.save('audio.mp3')
    print("Bot: " + res)
    print()
    playsound('audio.mp3')
    os.remove("audio.mp3")'''
    if tag=="goodbye":
        Flag=False
    return res

#chat()