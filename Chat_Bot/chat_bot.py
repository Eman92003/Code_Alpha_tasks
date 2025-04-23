##1- importing libraries
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from nltk.tokenize import TreebankWordTokenizer


lemmatizer = WordNetLemmatizer()
##reading the created files (words, classes) and json files
intents = json.loads(open(r'C:\Users\Eman Yaser\Documents\Code Alpha intern\chat bot\venv\intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbot.h5')

##creating a function that will be responsible for preprocess the sentence to generate the answer
def clean_up_sentence(sentence):
    tokenizer = TreebankWordTokenizer()
    sentence_words = tokenizer.tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

##creating a function that takes the output of cleanup function and creates a BOW and return it as numpy array
def BOW (sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

##Create a function for passing the created BOW to the pretrained model
def predict_class (sentence):
    bow = BOW(sentence)
    res = model.predict(np.array([bow]))[0]
    ##this is used for returning classes that have higher probabilities that will be used for generating responses
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    #sorting the choosen classes form highest to lowest ans append them to a list in the form of class and its correspondong probability
    results.sort(key= lambda x:x[1], reverse= True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

##creatin a function for generating answers
##it generate answers by using classes predicted and the basic json file and generate renadom answer from them
def get_response(intents_list, intents_json):
    ##all intents definition from basic json file
    list_of_intents = intents_json['intents']
    ##intent with highest probability
    tag = intents_list[0]['intent']
    ##match the predicted intent with the basic intent and choose one renadom response of it and return it as an answer
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print('Great!bot is running!!')

##printing the chatbot answer
while True:
    message = input('')
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)