import pickle
import tkinter
import sv_ttk
# Importing the required functions from sentiment.py
from sentiment import Sentiment


sentiment = Sentiment()

model = pickle.load(open('SentimentModel.pkl', 'rb'))
vectorizer = pickle.load(open('Vectorizer.pkl', "rb"))

inp = input('Enter your sentence: ')
res = sentiment.data_cleaning(inp)

document = vectorizer.transform([inp])
prediction = model.predict(document)

match int(prediction[0]):
    case 1:
        print('Very Negative Sentiment')
    case 2:
        print('Negative Sentiment')
    case 3:
        print('Neutral Sentiment')
    case 4:
        print('Positive Sentiment')
    case 5:
        print('Very Positive Sentiment')

root = tkinter.Tk()

e = tkinter.Entry(root, width=10)
e.pack()
e.insert(0, "Enter your sentence here: ")


root.mainloop()
