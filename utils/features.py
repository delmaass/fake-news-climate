# %% [markdown]
# # Features - Fake news detection

# %%
import spacy
import numpy as np
import pandas as pd
import pickle

from sklearn import metrics
from sklearn.svm import SVC
from spacy_readability import Readability
from spacy.language import Language

nlp = spacy.load("fr_dep_news_trf")

def get_readability(nlp, name):
    read = Readability()
    return read

Language.factory("my_readability", func=get_readability)
nlp.add_pipe("my_readability", last=True)

# %% [markdown]
# ## Dataset

# %%
def remove_useless(paragraph_ann):
    content = paragraph_ann["content"]
    paragraphs_to_delete = []
    entities_to_delete = []

    for index, paragraph in enumerate(content) :
        if 'label' in paragraph.keys() and paragraph["label"]==0:
            paragraphs_to_delete.append(index)

        else :
            if type(paragraph["content"])==list:
              for index_entity, entity_content in enumerate(paragraph["content"]):
                if type(entity_content) == dict and 'label' in entity_content.keys() and entity_content["label"]==0:
                      entities_to_delete.append((index, index_entity))

    for i in range(-1, -len(entities_to_delete)-1, -1) :
      index, index_entity = entities_to_delete[i]
      del paragraph_ann['content'][index]['content'][index_entity]

    for index in reversed(paragraphs_to_delete):
      del paragraph_ann['content'][index]

    return paragraph_ann


def fusion(paragraph_ann):
  label = paragraph_ann['label']
  author = paragraph_ann['author']
  title = paragraph_ann['title']
  date = paragraph_ann['date']
  content = ""
  for paragraph in paragraph_ann['content']:
    if type(paragraph['content']) == str :
      content += paragraph['content']
    elif type(paragraph['content']) == list :
      for entity in paragraph['content']:
        content += entity['content']
  json = {
    'label' : label, 
    'date': date,
    'title': title,
    'author': author,
    'content' : content,
  }
  return json

# %%
# assign directory
# ANNOTATIONS_FOLDER = 'C:\\Users\\louis\\Desktop\\NLP\\fake_news\\annotations'
# dataset = []
 
# # iterate over files in
# # that directory
# for filename in os.listdir(ANNOTATIONS_FOLDER):
#     f = os.path.join(ANNOTATIONS_FOLDER, filename)
#     # checking if it is a file
#     if os.path.isfile(f):
#         with open(f, 'r') as file:
#             data = json.load(file)
#             if data["label"] > 0:
#                 data = fusion(remove_useless(data))
#                 dataset += [(data["label"], data["content"])]

# dataset = np.array(dataset, dtype=str)

# PATH = 'text_dataset.csv'
# fields = ['label', 'article']
# dataset = np.array(pd.read_csv(PATH, usecols=fields), dtype=str)

# print(f'Dataset: {dataset.shape}')

labels = np.array(pickle.load(open("datasets/docs/labels_train.p", "rb")), dtype=object)
docs = np.array(pickle.load(open("datasets/docs/docs_train.p", "rb")), dtype=object)

train_dataset = np.vstack((labels, docs)).T

np.random.shuffle(train_dataset)

print(f'Dataset: {train_dataset.shape}')

# %%
# y_train, x_train_txt = np.array(train[:,0], dtype=int), [str(txt) for txt in train[:,1]]
# y_test, x_test_txt = np.array(test[:, 0], dtype=int), [str(txt) for txt in test[:,1]]
# print(f'Train: {len(x_train_txt)} ({type(x_train_txt[0])})')
# print(f'Test: {len(x_test_txt)}')
y_train, x_train_doc = np.array(train_dataset[:,0], dtype=int) - 1, list(train_dataset[:, 1])
y_test, x_test_doc = np.array(pickle.load(open("datasets/docs/labels_test.p", "rb")), dtype=int) - 1, pickle.load(open("datasets/docs/docs_test.p", "rb"))
y_val, x_val_doc = np.array(pickle.load(open("datasets/docs/labels_validation.p", "rb")), dtype=int) - 1, pickle.load(open("datasets/docs/docs_validation.p", "rb"))

print(f'Train: {len(x_train_doc)} ({type(x_train_doc[0])})')
print(f'Test: {len(x_test_doc)}')
print(f'Test: {len(x_val_doc)}')

# %%
# x_train_doc = list(nlp.pipe(x_train_txt))
# x_test_doc = list(nlp.pipe(x_test_txt))

# %% [markdown]
# ## Features

# %% [markdown]
# ### Functions

# %%
def get_punct_ratio(doc) -> float:
    n_token = len(doc)
    if n_token:
        return sum([1 if token.pos_ == "PUNCT" else 0 for token in doc]) / n_token
    else:
        return .0

def get_adv_ratio(doc) -> float:
    n_token = len(doc)
    if n_token:
        return sum([1 if token.pos_ == "ADV" else 0 for token in doc]) / n_token
    else:
        return .0

def get_fin_ratio(doc) -> float:
    n_token = len(doc)
    if n_token:
        return sum([1 if "Fin" in token.morph.get("VerbForm") else 0 for token in doc]) / n_token
    else:
        return .0   

def get_fkre(doc) -> float:
    return doc._.flesch_kincaid_reading_ease

def get_length(doc) -> float:
    return len(doc)

def get_expr(doc) -> float:
    return sum([1 if token.lemma_ in ['?', '!', '(', ')'] else 0 for token in doc])

# %%
FEATURES = {
    "PUNCT": get_punct_ratio,
    "ADV": get_adv_ratio,
    "FIN": get_fin_ratio,
    "FKRE": get_fkre,
    "LENGTH": get_length,
    "EXPR": get_expr,
}

features_train = np.zeros((len(x_train_doc), len(FEATURES)), dtype=float)
features_test = np.zeros((len(x_test_doc), len(FEATURES)), dtype=float)
features_val = np.zeros((len(x_val_doc), len(FEATURES)), dtype=float)

# %% [markdown]
# ### Computation

# %%
print("Generating features for train...")
for doc_idx, doc in enumerate(x_train_doc):
    for feature_idx, feature in enumerate(FEATURES.keys()):
        get_feature = FEATURES[feature]
        features_train[doc_idx, feature_idx] = get_feature(doc)

# %%
print("Generating features for test...")
for doc_idx, doc in enumerate(x_test_doc):
    for feature_idx, feature in enumerate(FEATURES.keys()):
        get_feature = FEATURES[feature]
        features_test[doc_idx, feature_idx] = get_feature(doc)

print("Generating features for validation...")
for doc_idx, doc in enumerate(x_val_doc):
    for feature_idx, feature in enumerate(FEATURES.keys()):
        get_feature = FEATURES[feature]
        features_val[doc_idx, feature_idx] = get_feature(doc)

# %%
classifier = SVC(class_weight='balanced')
print("Fitting classifier")
classifier.fit(features_train, y_train)
print("Predicting test...")
y_predicted = np.array(classifier.predict(features_test), dtype=int)

# %% [markdown]
# ## Scores

# %%
def evaluate(predictions, labels, metric='report'):
    if metric == 'report':
        return metrics.classification_report(labels, predictions, zero_division=0)
    elif metric == 'matrix':
        return metrics.confusion_matrix(labels, predictions)

# %%
report = evaluate(y_predicted, y_test)
print(report)
matrix = evaluate(y_predicted, y_test, metric="matrix")
print(matrix)

print("Predicting validation...")
y_predicted = np.array(classifier.predict(features_val), dtype=int)
report = evaluate(y_predicted, y_val)
print(report)
matrix = evaluate(y_predicted, y_val, metric="matrix")
print(matrix)

