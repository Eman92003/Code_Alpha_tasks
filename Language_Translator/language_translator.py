##for GUI (bottons, inputs,....)
from tkinter import *
##style tkiker wedgits (colors, sizes,....)
from tkinter import ttk
## free python library for google translation API
from googletrans import Translator, LANGUAGES

root = Tk()
root.geometry('800x400')
root.resizable(0,0)
root.config(bg = 'ghost white')
root.title('Language Translator')
##title of the box
Label(root, text = "Language Translator", font = "Arial 20 bold",bg = 'ghost white').pack()
##label for the input data
Label(root,text="Enter Text:",font= 'arial 13 bold',bg = 'ghost white').place(x=30, y=90)

##adding input field to get data from user (single line)
Input_text = Entry(root, width= 100)   ## 100 is maximum number of characters
Input_text.place(x=130, y=95)
Input_text.get()

##output label
Label(root,text="Output",font= 'arial 13 bold',bg = 'ghost white').place(x=30, y=200)
#output field (multiline feild)
#hight = #lines, width= #chars per line, wrap to fit the text if it is too long
output_text = Text(root, font='arial 12', height=5, wrap= WORD, padx=5, pady=5, width=60)
output_text.place(x=130, y=200)


## create a combo box to choose the destination language
language = list(LANGUAGES.values())  ##get all languages
##creating the combo box 
dest_lang = ttk.Combobox(root, values= language, width= 22)
dest_lang.place(x = 130, y = 150)
##default value in combobox
dest_lang.set('Choose langauge')


##Translation function
def translate():
    ##object from google translatort tool
    translator = Translator()
    ##use teanslate function to translate user input to the destination language choosen
    translated = translator.translate(text = Input_text.get(), dest=dest_lang.get())
    ##this line clear the previous text before adding the new translated one
    output_text.delete(1.0, END)
    ##adding the translated text to the output feild
    output_text.insert(END, translated.text)

##creating bottom for translating  
trans_btn = Button(root, text='Translate', font='arial 10 bold', pady= 5, command= translate, bg='#808080', activebackground='lightblue') ##command her to refer to the used function
trans_btn.place(x=600, y=140)

root.mainloop()