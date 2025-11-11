import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# load pre-trained DistilBERT sequence classification model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# TODO: load dataset and tokenize appropriately

# TODO: fine-tune training of model

# placeholder code below
text = 'Yeah right!'
inputs = tokenizer(text, return_tensors='pt')

outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1).item()

print('predictions: ', predictions)