import torch
from transformers import AutoTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# training checkpoint path
# checkpoint_path = f'files/SarcasmClassifierModel/checkpoint-2138' 

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
# tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

# load pre-trained DistilBERT sequence classification model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', id2label={0:'NEG',1:'POS'},label2id={'NEG':0,'POS':1})
# model = DistilBertForSequenceClassification.from_pretrained(checkpoint_path)

# load training dataset
data_total = pd.read_json('files/Sarcasm_Headlines_Dataset.json', lines=True)

# data cleaning

# drop "article_link" column
data_total = data_total.drop(data_total.columns[0], axis=1)

# rename "is_sarcastic" column to "label"
data_total.rename(columns={'is_sarcastic': 'label'}, inplace=True)

print(data_total)

# split the data to be 64% training, 16% validation and 20% test data
data_train_and_val, data_test = train_test_split(data_total, test_size=0.2, random_state=42)
data_train, data_val = train_test_split(data_train_and_val, test_size=0.2, random_state=42)

print(data_train.shape, data_val.shape, data_test.shape)

# tokenize datasets
tr_tok = tokenizer(data_train['headline'].tolist(), return_tensors='pt', truncation=True, padding=True, max_length=128)
val_tok = tokenizer(data_val['headline'].tolist(), return_tensors='pt', truncation=True, padding=True, max_length=128)
test_tok = tokenizer(data_test['headline'].tolist(), return_tensors='pt', truncation=True, padding=True, max_length=128)

# add tokenized outputs as new columns in dfs
data_train['input_ids'] = tr_tok['input_ids'].tolist()
data_train['attention_mask'] = tr_tok['attention_mask'].tolist()

data_val['input_ids'] = val_tok['input_ids'].tolist()
data_val['attention_mask'] = val_tok['attention_mask'].tolist()

data_test['input_ids'] = test_tok['input_ids'].tolist()
data_test['attention_mask'] = test_tok['attention_mask'].tolist()

# convert to Hugging Face datasets
dataset_train = Dataset.from_pandas(data_train)
dataset_val = Dataset.from_pandas(data_val)
dataset_test = Dataset.from_pandas(data_test)

# fine-tune training of model

# set training arguments
training_args = TrainingArguments(
    output_dir='files/SarcasmClassifierModel',
    learning_rate=2e-5,
    per_device_train_batch_size=16, ### TODO: Make batch size larger for faster training?
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    push_to_hub=False, # do not intend to upload model to Hugging Face during training
    fp16=True,
)

# define compute metrics
def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)

    # labels = pred.label_ids
    # preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }


# initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    # processing_class=tokenizer,
    # data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# train model
trainer.train()
# trainer.train(resume_from_checkpoint=checkpoint_path)

# evaluate model
trainer.evaluate()

# save model and tokenizer
model_save_path ='files/TrainedSarcasmClassifierModel'
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
    

# placeholder code below
text = 'Yeah right!'
inputs = tokenizer(text, return_tensors='pt')

outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=-1).item()

print('predictions: ', predictions)