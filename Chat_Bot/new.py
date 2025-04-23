#in this file we are using the json file to get components used in making BOW
#the json file here is working as the dataset will used to answer user questions
#it will contain intents (main tag) and inside it it has other tags (greeting, goodbye,...) 
#each of the internal tages has patterns (what possible words user can use) amd responses (what the chatbot will response)

##Importing libraries
import numpy as np
import tensorflow as tf
import random
import json
import pickle
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
#from nltk.tokenize import word_tokenize
from nltk.tokenize import TreebankWordTokenizer

##initilize lemmatizer
lemmatizer = WordNetLemmatizer()

##loading data from json file and option is read (just reading data from file)
intents = json.loads(open('intents.json').read())

words = [] #store the words from file (tokenized and lemmatized) 
classes = [] #store tag name in intents (greeting,....)
documents = [] #store tokenized answers and its corresponding tag
ignoreLetters = ['?','!','.',','] #what we will ignore (ex: panctuation marks...)

##preprocessing
#extracting words, classes, [words,class] in json file
for intent in intents['intents']: #main tag in json file
    for pattern in intent['patterns']:
        #wordList = word_tokenize(pattern, language='english')  #words of patterns in each tag
        tokenizer = TreebankWordTokenizer()
        wordList = tokenizer.tokenize(pattern)
        words.extend(wordList)  #adding tokenized words in all patterns in word list 
        documents.append((wordList, intent['tag']))  #ex: [['hi','there'],'greeting]
        if intent['tag'] not in classes:
            classes.append(intent['tag']) 

#now after having all words, lemmatize them, remove ignore letter
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]

##create BOW
#removing duplicates in words and classes and sorting them
words = sorted(set(words))
classes = sorted(set(classes))

##storing words and c;asses as files using pickle
pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

##creating list for storing BOW for each sentence, and output vector (class tag)
training =[]
 
#creating list will be used for creating output vector (class vector)
outputEmpty = [0] * len(classes)

##we will use document list as it contain each sentence and it class (what we need)
for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    #Create BOW vector for each pattern
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)
    
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1 #make as one hor encoder and put 1 on the place of the class 
    ## adding the BOW vector a
    # nd output vector together
    training.append(bag + outputRow)


##training model
#shuffeling the created data
random.shuffle(training)
#converting it to numpyarray and split it into x(BOW) , y(label vector)
training = np.array(training)
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

#building sequential model of 3 layers 
#the first layer with 128 neuron as after building BOW it will contain 128 value
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
##softmax is used to generate probabilistic output to generate responses natural and coherent according to classes
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
hist = model.fit(np.array(trainX), np.array(trainY), epochs =200, batch_size =5, verbose=1)
model.save('chatbot.h5',hist)

print('Excuted')
