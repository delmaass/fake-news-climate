from sklearn.model_selection import train_test_split
import spacy
import pickle
import numpy as np
import pandas as pd
from spacy_readability import Readability
from spacy.language import Language


PATH = 'text_dataset.csv'
fields = ['article id','label', 'article']
dataset = np.array(pd.read_csv(PATH, usecols=fields), dtype=str)


ids_train = np.array(pickle.load(open("ids_train.p", "rb")), dtype=int)
ids_test = np.array(pickle.load(open("ids_test.p", "rb")), dtype=int)
ids_val = np.array(pickle.load(open("ids_val.p", "rb")), dtype=int)



nlp = spacy.load("fr_dep_news_trf")

def get_readability(nlp, name):
    read = Readability()
    return read

Language.factory("my_readability", func=get_readability)
nlp.add_pipe("my_readability", last=True)


print(f'Dataset: {dataset.shape}')

texts = [str(txt) for txt in dataset[:, 2]]
docs = list(nlp.pipe(texts))
labels = [int(label) for label in dataset[:, 1]]
ids = [int(id_) for id_ in dataset[:,0]]

# Separate train, test and validation

train_docs = []
test_docs = []
val_docs = []

train_labels = []
test_labels = []
val_labels = []

for id, doc, label in zip(ids, docs, labels):
    if id in ids_train:
        train_docs.append(doc)
        train_labels.append(label)
    elif id in ids_test:
        test_docs.append(doc)
        test_labels.append(label)
    elif id in ids_val:
        val_docs.append(doc)
        val_labels.append(label)
    


pickle.dump(train_labels, open("labels_train.p", "wb"))
pickle.dump(train_docs, open("docs_train.p", "wb"))

pickle.dump(test_labels, open("labels_test.p", "wb"))
pickle.dump(test_docs, open("docs_test.p", "wb"))

pickle.dump(val_labels, open("labels_val.p", "wb"))
pickle.dump(val_docs, open("docs_val.p", "wb"))