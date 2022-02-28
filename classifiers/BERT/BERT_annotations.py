# %% [markdown]
# ## **Fine-tuning BERT for named-entity recognition**
# 
# Based on https://github.com/abhimishra91/transformers-tutorials

# %% [markdown]
# #### **Importing Python Libraries and preparing the environment**

# %%
# !pip install transformers seqeval[gpu]

# %%
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CamembertForTokenClassification, CamembertTokenizerFast
# from seqeval.metrics import classification_report

# %%
# from torch import cuda
# device = 'cuda' if cuda.is_available() else 'cpu'
device = 'cpu'
print(device)

# %% [markdown]
# #### **Downloading and preprocessing the data**
# Named entity recognition (NER) uses a specific annotation scheme, which is defined (at least for European languages) at the *word* level. An annotation scheme that is widely used is called **[IOB-tagging](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)**, which stands for Inside-Outside-Beginning. Each tag indicates whether the corresponding word is *inside*, *outside* or at the *beginning* of a specific named entity. The reason this is used is because named entities usually comprise more than 1 word. 
# 
# Let's have a look at an example. If you have a sentence like "Barack Obama was born in Hawaï", then the corresponding tags would be   [B-PERS, I-PERS, O, O, O, B-GEO]. B-PERS means that the word "Barack" is the beginning of a person, I-PERS means that the word "Obama" is inside a person, "O" means that the word "was" is outside a named entity, and so on. So one typically has as many tags as there are words in a sentence.
# 
# So if you want to train a deep learning model for NER, it requires that you have your data in this IOB format (or similar formats such as [BILOU](https://stackoverflow.com/questions/17116446/what-do-the-bilou-tags-mean-in-named-entity-recognition)). There exist many annotation tools which let you create these kind of annotations automatically (such as Spacy's [Prodigy](https://prodi.gy/), [Tagtog](https://docs.tagtog.net/) or [Doccano](https://github.com/doccano/doccano)). You can also use Spacy's [biluo_tags_from_offsets](https://spacy.io/api/goldparse#biluo_tags_from_offsets) function to convert annotations at the character level to IOB format.
# 
# Here, we will use a NER dataset from [Kaggle](https://www.kaggle.com/namanj27/ner-dataset) that is already in IOB format. One has to go to this web page, download the dataset, unzip it, and upload the csv file to this notebook. Let's print out the first few rows of this csv file:

# %%
data = pd.read_csv("word_labels_per_par.csv", encoding='unicode_escape')
data = data.drop(columns=['Unnamed: 0'])
data = data[data["word_labels"] != 'O']

array = data.to_numpy()
data.head()

# %% [markdown]
# Let's check how many sentences and words (and corresponding tags) there are in this dataset:

# %%
data.count()

# %%
tags = {}
for word_labels in array[:,1]:
    for c in word_labels.split(','):
        if c not in tags.keys():
            tags[c] = 1
        else:
            tags[c] += 1

print("Number of tags: {}".format(len(tags.keys())))
print(tags)

# %% [markdown]
# Let's suppose that '-' (too long annotations) are fake news (label 3). Lets also remove correct annotations (label 1)

# %%
# Replace long annotations by fake annotations
for idx in range(len(array[:, 1])):
    word_labels = array[idx, 1].split(',')
    previous_label = ''
    for label_idx, label in enumerate(word_labels):
        if label == '-':
            if previous_label != '-':
                word_labels[label_idx] = 'B-3'
            else:
                if label_idx < len(word_labels) - 1:
                    if word_labels[label_idx + 1] != '-':
                        word_labels[label_idx] = 'L-3'
                    else:
                        word_labels[label_idx] = 'I-3'
                else:
                    word_labels[label_idx] = 'L-3'
        previous_label = label
    array[idx, 1] = ','.join(word_labels)

# Remove correct annotations
CORRECT_LABELS = ['B-1', 'U-1', 'I-1', 'L-1']
for idx in range(len(array[:, 1])):
    word_labels = array[idx, 1].split(',')
    for label_idx, label in enumerate(word_labels):
        if label in CORRECT_LABELS:
            word_labels[label_idx] = 'O'
    array[idx, 1] = ','.join(word_labels)

tags = {}
for word_labels in array[:,1]:
    for c in word_labels.split(','):
        if c not in tags.keys():
            tags[c] = 1
        else:
            tags[c] += 1

print("Number of tags: {}".format(len(tags.keys())))
print(tags)

# %% [markdown]
# We create 2 dictionaries: one that maps individual tags to indices, and one that maps indices to their individual tags. This is necessary in order to create the labels (as computers work with numbers = indices, rather than words = tags) - see further in this notebook.

# %%
labels_to_ids = {k: v for v, k in enumerate(tags.keys())}
ids_to_labels = {v: k for v, k in enumerate(tags.keys())}
labels_to_ids

# %% [markdown]
# Let's replace the word_labels by the new ones

# %%
# USELESS : extract the annotated_paragraphs
annotated_paragraphs = []

for i, word_labels in enumerate(array[:, 1]):
    split = word_labels.split(',')
    for s in split:
        if s in list(labels_to_ids.keys())[1:]:
            annotated_paragraphs.append(array[i])
            break

annotated_paragraphs_df = pd.DataFrame(annotated_paragraphs, columns=["text", "word_labels"])
print(annotated_paragraphs_df.head())
# annotated_paragraphs_df.to_csv("annotated_paragraphs.csv")

# %%
# data["word_labels"] = array[:, 1]
# data.head()

# %%
# indexes = []

# print(len(array[:, 1]), len(data))

# for idx, word_labels in enumerate(array[:, 1]):
#     split = word_labels.split(',')
#     unique = list(set(split))
#     if len(unique) == 1:
#             indexes.append(idx)

# data = data.drop(data.index[indexes])

data = annotated_paragraphs_df

# %%
print(len(data))
print(data.head())

# %% [markdown]
# Let's verify that a random sentence and its corresponding tags are correct:

# %%
data.iloc[41].text

# %%
data.iloc[41].word_labels

# %% [markdown]
# #### **Preparing the dataset and dataloader**

# %% [markdown]
# Now that our data is preprocessed, we can turn it into PyTorch tensors such that we can provide it to the model. Let's start by defining some key variables that will be used later on in the training/evaluation process:

# %%
MAX_LEN = 256
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10
tokenizer = CamembertTokenizerFast.from_pretrained('camembert-base', do_lower_case=True)

# %% [markdown]
# A tricky part of NER with BERT is that BERT relies on **wordpiece tokenization**, rather than word tokenization. This means that we should also define the labels at the wordpiece-level, rather than the word-level! 
# 
# For example, if you have word like "Washington" which is labeled as "b-gpe", but it gets tokenized to "Wash", "##ing", "##ton", then one approach could be to handle this by only train the model on the tag labels for the first word piece token of a word (i.e. only label "Wash" with "b-gpe"). This is what was done in the original BERT paper, see Github discussion [here](https://github.com/huggingface/transformers/issues/64#issuecomment-443703063).
# 
# Note that this is a **design decision**. You could also decide to propagate the original label of the word to all of its word pieces and let the model train on this. In that case, the model should be able to produce the correct labels for each individual wordpiece. This was done in [this NER tutorial with BERT](https://github.com/chambliss/Multilingual_NER/blob/master/python/utils/main_utils.py#L118). Another design decision could be to give the first wordpiece of each word the original word label, and then use the label “X” for all subsequent subwords of that word. All of them seem to lead to good performance.
# 
# Below, we define a regular PyTorch [dataset class](https://pytorch.org/docs/stable/data.html) (which transforms examples of a dataframe to PyTorch tensors). Here, each sentence gets tokenized, the special tokens that BERT expects are added, the tokens are padded or truncated based on the max length of the model, the attention mask is created and the labels are created based on the dictionary which we defined above. Word pieces that should be ignored have a label of -100 (which is the default `ignore_index` of PyTorch's [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)).
# 
# For more information about BERT's inputs, see [here](https://huggingface.co/transformers/glossary.html). 
# 
# 
# 
# 
# 
# 
# 

# %%
class dataset(Dataset):
  def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

  def __getitem__(self, index):
        # step 1: get the sentence and word labels 
        sentence = self.data.text[index].strip().split()  
        word_labels = self.data.word_labels[index].split(",") 

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
                             is_split_into_words=True, 
                             return_offsets_mapping=True, 
                             padding='max_length', 
                             truncation=True, 
                             max_length=self.max_len)
        
        # step 3: create token labels only for first word pieces of each tokenized word
        labels = [labels_to_ids[label] for label in word_labels] 
        # code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
        # create an empty array of -100 of length max_length
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        
        # set only labels whose first offset position is 0 and the second is not 0
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
          if mapping[0] == 0 and mapping[1] != 0:
            # overwrite label
            label = labels[i]
            encoded_labels[idx] = label
            i += 1

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)
        
        return item

  def __len__(self):
        return self.len

# %% [markdown]
# Now, based on the class we defined above, we can create 2 datasets, one for training and one for testing. Let's use a 80/20 split:

# %%
train_size = 0.8
train_dataset = data.sample(frac=train_size,random_state=200)
test_dataset = data.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

print("FULL Dataset: {}".format(data.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = dataset(train_dataset, tokenizer, MAX_LEN)
testing_set = dataset(test_dataset, tokenizer, MAX_LEN)

# %% [markdown]
# Let's have a look at the first training example:

# %%
print(training_set[0])

# %% [markdown]
# Let's verify that the input ids and corresponding targets are correct:

# %%
for token, label in zip(tokenizer.convert_ids_to_tokens(training_set[0]["input_ids"]), training_set[0]["labels"]):
  print('{0:10}  {1}'.format(token, label))

# %% [markdown]
# Now, let's define the corresponding PyTorch dataloaders:

# %%
train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

# %% [markdown]
# #### **Defining the model**

# %% [markdown]
# Here we define the model, BertForTokenClassification, and load it with the pretrained weights of "bert-base-uncased". The only thing we need to additionally specify is the number of labels (as this will determine the architecture of the classification head).
# 
# Note that only the base layers are initialized with the pretrained weights. The token classification head of top has just randomly initialized weights, which we will train, together with the pretrained weights, using our labelled dataset. This is also printed as a warning when you run the code cell below.
# 
# Then, we move the model to the GPU.

# %%
model = CamembertForTokenClassification.from_pretrained('camembert-base', num_labels=len(labels_to_ids))
model.to(device)

# %% [markdown]
# #### **Training the model**
# 
# Before training the model, let's perform a sanity check, which I learned thanks to Andrej Karpathy's wonderful [cs231n course](http://cs231n.stanford.edu/) at Stanford (see also his [blog post about debugging neural networks](http://karpathy.github.io/2019/04/25/recipe/)). The initial loss of your model should be close to -ln(1/number of classes) = -ln(1/17) = 2.83. 
# 
# Why? Because we are using cross entropy loss. The cross entropy loss is defined as -ln(probability score of the model for the correct class). In the beginning, the weights are random, so the probability distribution for all of the classes for a given token will be uniform, meaning that the probability for the correct class will be near 1/17. The loss for a given token will thus be -ln(1/17). As PyTorch's [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) (which is used by `BertForTokenClassification`) uses *mean reduction* by default, it will compute the mean loss for each of the tokens in the sequence for which a label is provided. 
# 
# Let's verify this:
# 
# 

# %%
print(model.__dict__.keys())

# %%
inputs = training_set[2]
input_ids = inputs["input_ids"].unsqueeze(0).type(torch.LongTensor)
attention_mask = inputs["attention_mask"].unsqueeze(0).type(torch.LongTensor)
labels = inputs["labels"].unsqueeze(0).type(torch.LongTensor)

input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
labels = labels.to(device)

print(input_ids.size(), attention_mask.size(), labels.size())

outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
initial_loss = outputs[0]
print(initial_loss)

# %% [markdown]
# This looks good. Let's also verify that the logits of the neural network have a shape of (batch_size, sequence_length, num_labels):

# %%
tr_logits = outputs[1]
print(tr_logits.shape)

# %% [markdown]
# Next, we define the optimizer. Here, we are just going to use Adam with a default learning rate. One can also decide to use more advanced ones such as AdamW (Adam with weight decay fix), which is [included](https://huggingface.co/transformers/main_classes/optimizer_schedules.html) in the Transformers repository, and a learning rate scheduler, but we are not going to do that here.

# %%
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# %% [markdown]
# Now let's define a regular PyTorch training function. It is partly based on [a really good repository about multilingual NER](https://github.com/chambliss/Multilingual_NER/blob/master/python/utils/main_utils.py#L344).

# %%
# Defining the training function on the 80% of the dataset for tuning the bert model
def train(epoch):
    tr_loss, tr_accuracy = 0., 0.
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    # put model in training mode
    model.train()
    
    for idx, batch in enumerate(training_loader):
        
        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        labels = batch['labels'].to(device, dtype = torch.long)

        loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels, return_dict=False)
        tr_loss += loss

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)
        
        if idx % 100==0:
            loss_step = tr_loss/nb_tr_steps
            print(f"Training loss per 100 training steps: {loss_step}")
           
        # compute training accuracy
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
        
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        
        tr_labels.extend(labels)
        tr_preds.extend(predictions)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
    
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=MAX_GRAD_NORM
        )
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")

# %% [markdown]
# And let's train the model!

# %%
for epoch in range(EPOCHS):
    print(f"Training epoch: {epoch + 1}")
    train(epoch)

# %% [markdown]
# #### **Evaluating the model**

# %% [markdown]
# Now that we've trained our model, we can evaluate its performance on the held-out test set (which is 20% of the data). Note that here, no gradient updates are performed, the model just outputs its logits. 

# %%
def valid(model, testing_loader):
    # put model in evaluation mode
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []
    
    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            
            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)
            
            loss, eval_logits = model(input_ids=ids, attention_mask=mask, labels=labels, return_dict=False)
            
            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)
        
            if idx % 100==0:
                loss_step = eval_loss/nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")
              
            # compute evaluation accuracy
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
            
            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        
            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
            eval_labels.extend(labels)
            eval_preds.extend(predictions)
            
            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    labels = [ids_to_labels[id.item()] for id in eval_labels]
    predictions = [ids_to_labels[id.item()] for id in eval_preds]
    
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    return labels, predictions

# %% [markdown]
# As we can see below, performance is quite good! Accuracy on the test test is > 93%.

# %%
labels, predictions = valid(model, testing_loader)

# %% [markdown]
# However, the accuracy metric is misleading, as a lot of labels are "outside" (O), even after omitting predictions on the [PAD] tokens. What is important is looking at the precision, recall and f1-score of the individual tags. For this, we use the seqeval Python library: 

# %%
print(classification_report(labels, predictions))

# %% [markdown]
# Performance already seems quite good, but note that we've only trained for 1 epoch. An optimal approach would be to perform evaluation on a validation set while training to improve generalization.

# %% [markdown]
# #### **Saving the model for future use**

# %% [markdown]
# Finally, let's save the vocabulary (.txt) file, model weights (.bin) and the model's configuration (.json) to a directory, so that both the tokenizer and model can be re-loaded using the `from_pretrained()` class method.
# 

# %%
import os

directory = "./model"

if not os.path.exists(directory):
    os.makedirs(directory)

# save vocabulary of the tokenizer
tokenizer.save_vocabulary(directory)
# save the model weights and its configuration file
model.save_pretrained(directory)
print('All files saved')


