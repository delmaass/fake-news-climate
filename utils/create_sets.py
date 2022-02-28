import pickle
import numpy as np

features = np.array(pickle.load(open("features.p", "rb")), dtype=object)
labels = np.array(pickle.load(open("labels.p", "rb")), dtype=object)
articles = np.array(pickle.load(open("docs.p", "rb")), dtype=object)

split_border_1 = int(len(labels)*0.7)
split_border_2 = int(len(labels)*0.85)

articles_train, articles_test, articles_validation = articles[:split_border_1], articles[split_border_1:split_border_2], articles[split_border_2:]
features_train, features_test, features_validation = features[:split_border_1], features[split_border_1:split_border_2], features[split_border_2:]
labels_train, labels_test, labels_validation = labels[:split_border_1], labels[split_border_1:split_border_2], labels[split_border_2:]

pickle.dump(articles_train, open("docs_train.p", "wb"))
pickle.dump(articles_test, open("docs_test.p", "wb"))
pickle.dump(articles_validation, open("docs_val.p", "wb"))

pickle.dump(features_train, open("features_train.p", "wb"))
pickle.dump(features_test, open("features_test.p", "wb"))
pickle.dump(features_validation, open("features_val.p", "wb"))

pickle.dump(labels_train, open("labels_train.p", "wb"))
pickle.dump(labels_test, open("labels_test.p", "wb"))
pickle.dump(labels_validation, open("labels_val.p", "wb"))