import matplotlib.pyplot as plt
import numpy as np
import os, json
import pickle
import spacy
from spacy_readability import Readability
from spacy.language import Language


nlp = spacy.load("fr_dep_news_trf")

def get_readability(nlp, name):
    read = Readability()
    return read

Language.factory("my_readability", func=get_readability)
nlp.add_pipe("my_readability", last=True)
SAVE_PATH = 'feature_plots/'

labels = np.array(pickle.load(open("labels_par.p", "rb")), dtype=object)
docs = np.array(pickle.load(open("paragraphs.p", "rb")), dtype=object)

dataset = np.vstack((labels, docs)).T

labels_core = np.array(pickle.load(open("labels_core.p", "rb")), dtype=object)
docs_core = np.array(pickle.load(open("docs_core.p", "rb")), dtype=object)

dataset_core = np.vstack((labels_core, docs_core)).T

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

def get_quoations(doc) -> float:
    return sum([1 if token.is_quote else 0 for token in doc])

def get_number_ratio(doc) -> float:
    n_token = len(doc)
    if n_token:
        return sum([1 if token.like_num else 0 for token in doc]) / n_token
    else:
        return .0

def get_oov_ratio(doc) -> float: # vocabulary may be empty
    n_token = len(doc)
    if n_token:
        return sum([1 if token.is_oov else 0 for token in doc]) / n_token
    else:
        return .0

def get_sentiment_mean(doc) -> float: # sentimentt = 0 for every word in the current pipeline
    n_token = len(doc)
    if n_token:
        return sum([token.sentiment for token in doc]) / n_token
    else:
        return .0

def get_fake_similarity(doc, fake_tokens) -> float:
    pass

def get_parsing_density(doc) -> float:
    n_token = len(doc)
    if n_token:
        return sum([len([t.text for t in token.subtree]) for token in doc]) / n_token
    else:
        return .0

def get_sentences_lengths(doc) -> float:
    n_sents = len([sent for sent in doc.sents])
    if n_sents:
        return sum([len(sent) for sent in doc.sents]) / n_sents
    else:
        return .0

def get_conditional(doc) -> float:
    n_token = len(doc)
    if n_token:
        return sum([1 if "Cnd" in token.morph.get("Mood") else 0 for token in doc]) / n_token
    else:
        return .0
        
def get_first_person(doc) -> float:
    n_token = len(doc)
    if n_token:
        return sum([1 if "1" in token.morph.get("Person") else 0 for token in doc]) / n_token
    else:
        return .0


def apply_feature(feature, data):
    values = []
    for text in data[:, 1]:
        values.append(feature(text))
    return np.array(values)

def save_features_bis(feature_list, data):
    features = np.empty((np.shape(data)[0]))
    for feature in feature_list:
        values = apply_feature(feature, data)
        print(np.shape(features.shape))
        print(np.shape(values))
        features = np.vstack((features, values))
        #features = np.concatenate((features, values.T), axis=1)
    features = features.T
    print(np.shape(features))
    print(features)
    pickle.dump(features, open("features_par.p", "wb"))
    
def save_features(feature_list, data):
    features = [apply_feature(feature, data) for feature in feature_list]
    array = np.concatenate(features, axis = 1)
    print(np.shape(array))
    pickle.dump(features, open("features.p", "wb"))



if __name__=='__main__':
    feature_list = [get_adv_ratio, get_quoations, get_fin_ratio, get_expr, get_first_person]
    save_features_bis(feature_list, dataset)