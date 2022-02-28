import numpy as np
import pandas as pd
from sklearn import metrics
import spacy
from extract_annotated_sentences import get_sentences_from_doc

from sentences_classification import load_model_tokenizer, predict

def classify_from_sentences_labels(predicted_labels):
    (unique, counts) = np.unique(predicted_labels, return_counts=True)
    count_labels = np.asarray((unique, counts)).T
    max_idx = np.argmax(count_labels[:, 1])
    return count_labels[max_idx][0]

if __name__ == "__main__":
    model, tokenizer = load_model_tokenizer()

    nlp = spacy.load("fr_core_news_md", exclude=["parser"])
    nlp.enable_pipe("senter")

    df = pd.read_csv("datasets/articles/test_text_dataset.csv", usecols=["label", "article"])
    n_rows = len(df)
    labels = df.label.values - 1
    contents = df.article.values

    predicted_labels = []

    for id, content in enumerate(contents):
        print(f'{id}/{n_rows}')
        doc = nlp(str(content))
        sentences = get_sentences_from_doc(doc)
        predicted_sentences_labels = predict(sentences, model, tokenizer)
        predicted_label = classify_from_sentences_labels(predicted_sentences_labels)
        predicted_labels.append(predicted_label)

    confusion_matrix = metrics.classification_report(labels, predicted_labels, zero_division=0)
    report = metrics.confusion_matrix(labels, predicted_labels)

    print("PREDICT ON TEST SET")
    print(report)
    print(confusion_matrix)
        
        
