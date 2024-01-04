import pickle
import tkinter
from reddit import RedditClass  # Importing the reddit.py file
from sentiment import Sentiment  # Importing the required functions from sentiment.py file
sentiment = Sentiment()

root = tkinter.Tk()
root.title("Sentiment Analysis Using RoBERTa")
root.geometry("400x200")


def button_press():
    tkinter.Label(root, text=var1.get())
    if var1.get() == 1:
        reddit = RedditClass()
        reddit.redditapi()
        tkinter.Label(root, text="Reddit data has been run").grid()
    if var2.get() == 1:
        sentiment.run_sentiments()
        tkinter.Label(root, text="Sentiment has been assigned to the data").grid()
    if var3.get() == 1:
        model, vectorizer, train_acc, test_acc = sentiment.run_model()

        def save_model():
            pickle.dump(model, open('SentimentModel.pkl', 'wb'))
            pickle.dump(vectorizer, open('Vectorizer.pkl', 'wb'))
            tkinter.Label(root, text="Model has been saved").grid()

        tkinter.Label(root, text="Model has been trained").grid()
        custom_dialog = tkinter.Toplevel(root)
        custom_dialog.title('Save Model')
        tkinter.Label(custom_dialog, text=f"Training data accuracy: {train_acc}").grid()
        tkinter.Label(custom_dialog, text=f"Test data accuracy: {test_acc}").grid()
        tkinter.Label(custom_dialog, text="Do you want to save the model?").grid()
        tkinter.Button(custom_dialog, text="Yes", command=save_model).grid()
        tkinter.Button(custom_dialog, text="No", command=custom_dialog.destroy).grid()


var1 = tkinter.IntVar()
var2 = tkinter.IntVar()
var3 = tkinter.IntVar()

tkinter.Checkbutton(root, text="Get Reddit Data", variable=var1).grid(row=0, column=1)
tkinter.Checkbutton(root, text="Get Sentiment Data", variable=var2).grid(row=1, column=1)
tkinter.Checkbutton(root, text="Train the model", variable=var3).grid(row=2, column=1)

tkinter.Button(root, text="Run", command=button_press).grid(row=4, column=1)

e = tkinter.Entry(root, width=20)
e.grid(row=1, column=0)
e.insert(0, "Enter your sentence here: ")


def button_press_2():
    print(e.get())
    model = pickle.load(open('SentimentModel.pkl', 'rb'))
    vectorizer = pickle.load(open('Vectorizer.pkl', "rb"))

    inp = e.get()
    print('Data Received')
    res = sentiment.data_cleaning(inp)
    print('Data Cleaned')
    document = vectorizer.transform([res])
    prediction = model.predict(document)
    print('Prediction Made')
    match int(prediction[0]):
        case 1:
            tkinter.Label(root, text="Negative Sentiment").grid()
        case 2:
            tkinter.Label(root, text="Neutral Sentiment").grid()
        case 3:
            tkinter.Label(root, text="Positive Sentiment").grid()
    print('Done')


tkinter.Button(root, text="Get Sentiment", command=button_press_2).grid(row=2, column=0)

root.mainloop()
