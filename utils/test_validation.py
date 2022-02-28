import os
import json
import time
import torch
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import pickle

import seaborn
from sklearn import metrics
from torch.utils.data import TensorDataset, random_split, \
                            DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertForSequenceClassification, CamembertTokenizer, AutoConfig, AutoModel, \
                         AdamW, get_linear_schedule_with_warmup


docs_val = np.array(pickle.load(open("docs_val.p", "rb")), dtype=object)
features_val = np.array(pickle.load(open("features_val.p", "rb")), dtype=np.float32)
labels_val = np.array(pickle.load(open("labels_val.p", "rb")), dtype=int)-1

num_extra_dims = np.shape(features_val)[1]
num_labels = len(set(labels_val))


TOKENIZER = CamembertTokenizer.from_pretrained(
    'camembert-base',
    do_lower_case=True)

def preprocess_spacy(docs, pos=["PUNCT", "ADV", "ADJ", "VERB", "NOUN"]):
    texts = [" ".join([token.text for token in doc if not token.is_stop and token.pos_ in pos]) for doc in docs]

    return texts

def preprocess(raw_articles, features = None, labels=None):
    """
        Create pytorch dataloader from raw data
    """

    # https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_encode_plus.truncation

    encoded_batch = TOKENIZER.batch_encode_plus(raw_articles,
                                                add_special_tokens=False,
                                                padding = True,
                                                truncation = True,
                                                max_length = 512,
                                                return_attention_mask=True,
                                                return_tensors = 'pt')
        

    if features is not None:
        features = torch.tensor(features)
        if labels is not None:
            labels = torch.tensor(labels)
            return encoded_batch['input_ids'], encoded_batch['attention_mask'], features, labels
        return encoded_batch['input_ids'], encoded_batch['attention_mask'], features
    
    else :
        if labels is not None:
            labels = torch.tensor(labels)
            return encoded_batch['input_ids'], encoded_batch['attention_mask'], labels
        return encoded_batch['input_ids'], encoded_batch['attention_mask']
        

articles_validation = preprocess_spacy(docs_val)

input_ids, attention_mask, features_validation, labels_validation = preprocess(docs_val, features_val, labels_val)


# articles = preprocess_spacy(docs)

"""# Split train-validation
split_border = int(len(labels)*0.8)
articles_train, articles_validation = articles[:split_border], articles[split_border:]
features_train, features_validation = features[:split_border], features[split_border:]
labels_train, labels_validation = labels[:split_border], labels[split_border:]"""

# Combine the training inputs into a TensorDataset

class CustomModel(torch.nn.Module):
    """
    This takes a transformer backbone and puts a slightly-modified classification head on top.
    
    """

    def __init__(self, model_name, num_extra_dims, num_labels):
        # num_extra_dims corresponds to the number of extra dimensions of numerical/categorical data

        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        num_hidden_size = self.transformer.config.hidden_size # May be different depending on which model you use. Common sizes are 768 and 1024. Look in the config.json file 
        self.classifier = torch.nn.Linear(num_hidden_size+num_extra_dims, num_labels)


    def forward(self, input_ids, extra_data, attention_mask=None):
        """
        extra_data should be of shape [batch_size, dim] 
        where dim is the number of additional numerical/categorical dimensions
        """

        hidden_states = self.transformer(input_ids=input_ids, attention_mask=attention_mask) # [batch size, sequence length, hidden size]

        cls_embeds = hidden_states[0][:, 0, :] # [batch size, hidden size]

        concat = torch.cat((cls_embeds, extra_data), dim=-1) # [batch size, hidden size+num extra dims]

        output = self.classifier(concat) # [batch size, num labels]

        return output

PATH = "customed_camembert_model#2.model"
model_name = "camembert-base"
model = CustomModel(model_name, num_extra_dims=num_extra_dims, num_labels=num_labels)
model = model.load_state_dict(torch.load(PATH))
model.eval()

def predict(articles, features, model=model):
    with torch.no_grad():
        model.eval()
        input_ids, attention_mask, extra_data = preprocess(articles, features)
        output = model(input_ids, extra_data, attention_mask=attention_mask)
        return torch.argmax(output, dim=1)

def evaluate(articles, features, labels, metric='report'):
    predictions = predict(articles, features)
    if metric == 'report':
        return metrics.classification_report(labels, predictions, zero_division=0)
    elif metric == 'matrix':
        return metrics.confusion_matrix(labels, predictions)

confusion_matrix = evaluate(articles_validation, features_validation, labels_validation, 'matrix')
report = evaluate(articles_validation, features_validation, labels_validation, 'report')
print(report)
print(confusion_matrix)
seaborn.heatmap(confusion_matrix)