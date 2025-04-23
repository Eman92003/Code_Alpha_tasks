import tkinter as tk
from tkinter import scrolledtext
import random
import json
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from nltk.tokenize import TreebankWordTokenizer

# --- Load essentials ---
lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()

intents = json.loads(open(r'C:\Users\Eman Yaser\Documents\Code Alpha intern\chat bot\venv\intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot.h5')

# --- Chatbot Functions ---
def clean_up_sentence(sentence):
    sentence_words = tokenizer.tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def BOW(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = BOW(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if not intents_list:
        return "I didn't understand. Try again."
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

# --- GUI Functions ---
def send_message():
    user_input = entry.get()
    if user_input.strip() == "":
        return
    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, f"You: {user_input}\n")
    entry.delete(0, tk.END)

    ints = predict_class(user_input)
    response = get_response(ints, intents)
    chat_log.insert(tk.END, f"Bot: {response}\n\n")
    chat_log.config(state=tk.DISABLED)
    chat_log.yview(tk.END)

# --- Tkinter GUI Setup ---
root = tk.Tk()
root.title("AI Chatbot")
root.geometry("500x500")

chat_log = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED)
chat_log.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

entry = tk.Entry(root, width=60)
entry.pack(side=tk.LEFT, padx=(10, 0), pady=(0, 10))
entry.bind("<Return>", lambda event: send_message())  # Pressing Enter sends message

send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack(side=tk.RIGHT, padx=(0, 10), pady=(0, 10))

root.mainloop()
