from sklearn.model_selection import train_test_split
import spacy
import pickle
import numpy as np
import pandas as pd

from spacy_readability import Readability
from spacy.language import Language

nlp = spacy.load("fr_dep_news_trf")
#nlp = spacy.load("fr_core_news_md")

def get_readability(nlp, name):
    read = Readability()
    return read

Language.factory("my_readability", func=get_readability)
nlp.add_pipe("my_readability", last=True)

PATH = 'text_dataset.csv'
fields = ['label', 'article']
dataset = np.array(pd.read_csv(PATH, usecols=fields), dtype=str)

print(f'Dataset: {dataset.shape}')

texts = [str(txt) for txt in dataset[:, 1]]
docs = list(nlp.pipe(texts))
labels = [int(label) for label in dataset[:, 0]]

pickle.dump(labels, open("labels.p", "wb"))
pickle.dump(docs, open("docs.p", "wb"))