# https://spacy.io/api/token

import matplotlib.pyplot as plt
import numpy as np
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

labels = np.array(pickle.load(open("labels.p", "rb")), dtype=object)
docs = np.array(pickle.load(open("docs.p", "rb")), dtype=object)

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
    true_values = []
    biased_values = []
    fake_values = []
    for _, doc in enumerate(data):
        label = int(doc[0])
        text = doc[1]
        if label == 1 :
            true_values.append(feature(text))
        elif label == 2 :
            biased_values.append(feature(text))
        elif label == 3 :
            fake_values.append(feature(text))


    return np.array(true_values), np.array(biased_values), np.array(fake_values)    


def plot(feature, feature_name, data = dataset):

    true_values, biased_values, fake_values = apply_feature(feature, data)
    fig, axs = plt.subplots(3)
    fig.suptitle(feature_name)
    x_limit = [0, max(2*true_values.mean(), 2*fake_values.mean())]
    axs[0].hist(true_values, bins = 150, color = 'green')
    axs[0].set(ylabel='TRUE')
    axs[0].axvline(true_values.mean(), color='k', linestyle='dashed', linewidth=1)
    axs[0].set_xlim(x_limit)
    axs[1].hist(biased_values, bins = 150, color = 'pink')
    axs[1].set(ylabel='BIASED')
    axs[1].axvline(biased_values.mean(), color='k', linestyle='dashed', linewidth=1)
    axs[1].set_xlim(x_limit)
    axs[2].hist(fake_values, bins = 150, color = 'red')
    axs[2].set(ylabel='FAKE')
    axs[2].axvline(fake_values.mean(), color='k', linestyle='dashed', linewidth=1)
    axs[2].set_xlim(x_limit)
    path = SAVE_PATH + feature_name
    plt.savefig(path)

if __name__== '__main__':
    plot(get_punct_ratio, 'punctuation_ratio')
    plot(get_adv_ratio, 'adverb_ratio')
    plot(get_fin_ratio, 'modal_verbs_ratio_(fin)')
    plot(get_length, 'Number_of_spacy_tokens')
    plot(get_expr, '!_?_(_)')
    plot(get_fkre, 'fkre_readibility')
    plot(get_oov_ratio, 'oov_ratio', data=dataset_core)
    plot(get_parsing_density, 'parsing_density')
    plot(get_sentiment_mean, 'sentiment_mean', data=dataset_core)
    plot(get_quoations, 'number_of_quotations')
    plot(get_number_ratio, 'number_ratio')
    plot(get_sentences_lengths, 'sentences_lengths')
    plot(get_conditional, 'conditionnel_ratio')
    plot(get_first_person, 'first_person_ratio')
    # plot the other features as well !
