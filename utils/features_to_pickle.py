import numpy as np
import pickle

train_labels = np.array(pickle.load(open("labels_train.p", "rb")), dtype=object)
train_docs = np.array(pickle.load(open("docs_train.p", "rb")), dtype=object)
train_dataset = np.vstack((train_labels, train_docs)).T

test_labels = np.array(pickle.load(open("labels_test.p", "rb")), dtype=object)
test_docs = np.array(pickle.load(open("docs_test.p", "rb")), dtype=object)
test_dataset = np.vstack((test_labels, test_docs)).T

val_labels = np.array(pickle.load(open("labels_val.p", "rb")), dtype=object)
val_docs = np.array(pickle.load(open("docs_val.p", "rb")), dtype=object)
val_dataset = np.vstack((val_labels, val_docs)).T


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

def save_features_bis(feature_list, data, name):
    features = np.empty((np.shape(data)[0]))
    for feature in feature_list:
        values = apply_feature(feature, data)
        features = np.vstack((features, values))
    features = features.T
    pickle.dump(features, open(name, "wb"))
    
def save_features(feature_list, data):
    features = [apply_feature(feature, data) for feature in feature_list]
    array = np.concatenate(features, axis = 1)
    print(np.shape(array))
    pickle.dump(features, open("features.p", "wb"))



if __name__=='__main__':
    feature_list = [get_adv_ratio, get_quoations, get_fin_ratio, get_expr, get_first_person]
    save_features_bis(feature_list, train_dataset, "features_train.p")
    save_features_bis(feature_list, test_dataset, "features_test.p")
    save_features_bis(feature_list, val_dataset, "features_val.p")