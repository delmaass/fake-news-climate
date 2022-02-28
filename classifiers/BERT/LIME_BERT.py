# %%
from __future__ import print_function
import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
import pandas as pd
import spacy
from transformers import CamembertForSequenceClassification, CamembertTokenizer
import torch
from lime import lime_text
from lime.lime_text import LimeTextExplainer

np.random.seed(500)
nlp = spacy.load("fr_core_news_md")

# %%
def sigmoid(x):
  result = []
  for value in x:
    result += [1 / (1 + np.exp(-value))]
  return list(result / sum(result))

# %%
TRAIN_PATH = "datasets/articles/train_text_dataset.csv"
TEST_PATH = "datasets/articles/test_text_dataset.csv"

fields = ["label", "article"]

train_df = pd.read_csv(TRAIN_PATH, usecols=fields)
test_df = pd.read_csv(TEST_PATH, usecols=fields)

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

# x_train = preprocess([str(text) for text in train_df["article"].values])
# np.save("datasets/articles/x_train.npy", x_train)
x_train = np.load("datasets/articles/x_train.npy")

# x_test = preprocess([str(text) for text in test_df["article"].values])
# np.save("datasets/articles/x_test.npy", x_test)
x_test = np.load("datasets/articles/x_test.npy")

y_train, y_test = train_df["label"].values - 1, test_df["label"].values - 1

# %%
# making class names shorter
class_names = ["true", "biased", "fake"]
print(','.join(class_names))

# %%
MODEL_PATH = "models/first_camembert_model_tagtog.model"

def load_model_tokenizer():
    model = CamembertForSequenceClassification.from_pretrained(
        'camembert-base',
        num_labels = 3
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    tokenizer = CamembertTokenizer.from_pretrained(
        'camembert-base',
        do_lower_case=True
    )
    return model, tokenizer

model, tokenizer = load_model_tokenizer()

# %%
def process(raw_articles, labels=None):
    """
        Create pytorch dataloader from raw data
    """

    # https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_encode_plus.truncation

    encoded_batch = tokenizer.batch_encode_plus(raw_articles,
                                                add_special_tokens=False,
                                                padding = True,
                                                truncation = True,
                                                max_length = 512,
                                                return_attention_mask=True,
                                                return_tensors = 'pt')
        

    if labels:
        labels = torch.tensor(labels)
        return encoded_batch['input_ids'], encoded_batch['attention_mask'], labels
    return encoded_batch['input_ids'], encoded_batch['attention_mask']

def predict(articles):
    with torch.no_grad():
        input_ids, attention_mask = process(articles)
        output = model(input_ids, attention_mask=attention_mask)
        return torch.argmax(output[0], dim=1).tolist()

def predict_proba(articles):
    with torch.no_grad():
        input_ids, attention_mask = process(articles)
        output = model(input_ids, attention_mask=attention_mask)
        result = []
        for row in output[0].tolist():
            result.append(sigmoid(row))
        return np.array(result)

# %%
pred = predict(x_test)
print(sklearn.metrics.f1_score(y_test, pred, average='weighted'))

# %%
print(predict_proba([x_train[0]]))

# %%
explainer = LimeTextExplainer(class_names=class_names)

# %%

for idx in range(200):
    exp = explainer.explain_instance(x_train[idx], predict_proba, num_features=6, labels=[0, 1, 2])

    # %%
    # print('Document id: %d' % idx)
    # print('Predicted class =', class_names[predict(x_test[idx])[0]])
    # print('True class: %s' % class_names[y_test[idx]])

    # # %%
    # print ('Explanation for class %s' % class_names[0])
    # print ('\n'.join(map(str, exp.as_list(label=0))))
    # print ()
    # print ('Explanation for class %s' % class_names[1])
    # print ('\n'.join(map(str, exp.as_list(label=1))))
    # print ()
    # print ('Explanation for class %s' % class_names[2])
    # print ('\n'.join(map(str, exp.as_list(label=2))))

    # %%
    exp.save_to_file(f"explanations/{idx}.html")

