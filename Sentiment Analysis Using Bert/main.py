import pickle
import customtkinter
from reddit import RedditClass
from sentiment import Sentiment

sentiment = Sentiment()

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme('green')

root = customtkinter.CTk()
root.title('Sentiment Analysis with RoBERTa')
root.geometry("500x350")

frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=60, fill="both", expand=True)

def button_press ():
    if var1.get() == 1:
        reddit = RedditClass()
        reddit.redditapi()
        customtkinter.CTkLabel(frame, text="Reddit data has been run").pack()
    if var2.get() == 1:
        sentiment.run_sentiments()
        customtkinter.CTkLabel(frame, text="Sentiment has been assigned to the data").pack()
    if var3.get() == 1:
        model, vectorizer, train_acc, test_acc = sentiment.run_model()

        def save_model():
            pickle.dump(model, open('SentimentModel.pkl', 'wb'))
            pickle.dump(vectorizer, open('Vectorizer.pkl', 'wb'))
            customtkinter.CTkLabel(root, text="Model has been saved").pack()
            custom_dialog.destroy()

        customtkinter.CTkLabel(frame, text="Model has been trained").pack()
        custom_dialog = customtkinter.CTkToplevel(root)
        custom_dialog.title('Save Model')
        customtkinter.CTkLabel(custom_dialog, text=f"Training data accuracy: {train_acc}").pack()
        customtkinter.CTkLabel(custom_dialog, text=f"Test data accuracy: {test_acc}").pack()
        customtkinter.CTkLabel(custom_dialog, text="Do you want to save the model?").pack()
        customtkinter.CTkButton(custom_dialog, text="Yes", command=save_model).pack()
        customtkinter.CTkButton(custom_dialog, text="No", command=custom_dialog.destroy).pack()

var1 = customtkinter.IntVar()
var2 = customtkinter.IntVar()
var3 = customtkinter.IntVar()

customtkinter.CTkCheckBox(frame, text="Get Reddit Data", variable=var1).pack(pady=5, padx=10)
customtkinter.CTkCheckBox(frame, text="Get Sentiment Data", variable=var2).pack(pady=5, padx=10)
customtkinter.CTkCheckBox(frame, text="Train the model", variable=var3).pack(pady=5, padx=10)

customtkinter.CTkButton(frame, text="Run", command=button_press).pack(pady=6, padx=10)

e = customtkinter.CTkEntry(frame, width=200)
e.pack()
e.insert(0, "Enter your sentence here:")

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
        case 0:
            customtkinter.CTkLabel(frame, text="Negative Sentiment").pack()
        case 1:
            customtkinter.CTkLabel(frame, text="Neutral Sentiment").pack()
        case 2:
            customtkinter.CTkLabel(frame, text="Positive Sentiment").pack()
    print(int(prediction[0]))

customtkinter.CTkButton(frame, text="Get Sentiment", command=button_press_2).pack(pady=6, padx=10)

root.mainloop()
