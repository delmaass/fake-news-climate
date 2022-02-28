# %%
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

np.random.seed(500)
nlp = spacy.load("fr_core_news_md")

# %%
TRAIN_PATH = "datasets/articles/train_text_dataset.csv"
TEST_PATH = "datasets/articles/test_text_dataset.csv"

fields = ["label", "article"]

train_df = pd.read_csv(TRAIN_PATH, usecols=fields)
test_df = pd.read_csv(TEST_PATH, usecols=fields)

# %%
# Basic cleansing
def cleansing(doc):
    # Remove stop words
    doc = [token for token in doc if not token.is_stop]
    return doc

def keep_specific_pos(doc, pos=["ADV", "ADJ", "VERB", "NOUN"]):
    doc = [token for token in doc if token.pos_ in pos]
    return doc

def preprocess(data):
    docs = list(nlp.pipe(data))
    preprocess_docs = [keep_specific_pos(cleansing(doc)) for doc in docs]
    # Doc -> Text (+ lemmatization)
    output_texts = [" ".join([token.lemma_ for token in doc]) for doc in preprocess_docs]
    return output_texts

# %%
x_train = preprocess([str(text) for text in train_df["article"].values])

# %%
x_test = preprocess([str(text) for text in test_df["article"].values])

# %%
y_train, y_test = train_df["label"].values - 1, test_df["label"].values - 1

# %%
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(x_train + x_test)
Train_X_Tfidf = Tfidf_vect.transform(x_train)
Test_X_Tfidf = Tfidf_vect.transform(x_test)

# %%
print(Tfidf_vect.vocabulary_)

# %%
id_to_word = {v: k for k, v in Tfidf_vect.vocabulary_.items()}

# %%
print(Train_X_Tfidf)

# %%
# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,y_train)# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, y_test)*100)

# %%
for i in range(SVM.support_.shape[0]):
    label = SVM.predict(SVM.support_vectors_.getrow(i))
    word = id_to_word[SVM.support_vectors_.getrow(i).argmax()]
    print(f'{label}: {word}')

# %%
arg_max = SVM.support_vectors_.sum(0).argsort().transpose()

for i in range(1, len(arg_max) + 1):
    id = int(arg_max[-i])
    print(id_to_word[id])

# %%



