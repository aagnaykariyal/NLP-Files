import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

tokens = tokenizer.encode('I hate you', return_tensors='pt')
result = model(tokens)  # This line returns a tensor which consists of the probabilities of all the possibilities of
# sentiment from 1 to 5.
print(result)
'''From printing the result we can see that sentiment 1/bad sentiment has the probability of 3.1262 and is the highest 
    sentiment. Lets see how we can turn this into a use-able value'''
# print(int(torch.argmax(result.logits))+1)
'''Here we take the highest value and print the representative value by adding 1 to the position of the tensor'''
# print(int(torch.argmax(result.logits))+1)
