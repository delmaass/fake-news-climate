# %%
import time
import torch
import datetime
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.utils import class_weight
from torch.utils.data import TensorDataset, \
                            DataLoader, RandomSampler, SequentialSampler
from transformers import CamembertForSequenceClassification, CamembertTokenizer, \
                         AdamW, get_linear_schedule_with_warmup

# nlp = spacy.load("fr_dep_news_trf")
# Functions : preprocess() (create dataloaders from raw data) 
# load_models() (load tokenizers and models) training() (loop of one training step) evaluate()

# %%
TRAIN_SET_PATH = 'datasets/sentences/train_art_ann_sen.csv'
TEST_SET_PATH = 'datasets/sentences/test_art_ann_sen.csv'

fields = ['article_id', 'label', 'text']
train_df = pd.read_csv(TRAIN_SET_PATH, usecols=fields)
test_df = pd.read_csv(TEST_SET_PATH, usecols=fields)

print(f'Train: {train_df.shape}')
print(train_df.head(2))
print(train_df.tail(2))

print(f'Test: {test_df.shape}')
print(test_df.head(2))
print(test_df.tail(2))

train_df = train_df.sample(frac=1).reset_index(drop=True)

# %%
# 1 = true
# 2 = biased
# 3 = fake

# %%
#df['label'] = df['label'].replace([1], 0) # true label is now 0
#df['label'] = df['label'].replace([2, 3], 1) # fake and biased label are now 1

# %%
# labels = np.array(pickle.load(open("labels.p", "rb")), dtype=object)
# docs = np.array(pickle.load(open("docs.p", "rb")), dtype=object)

# dataset = np.vstack((labels, docs)).T

# np.random.shuffle(dataset)

TOKENIZER = CamembertTokenizer.from_pretrained(
    'camembert-base',
    do_lower_case=True)

# %%
# def preprocess_spacy(docs, pos=["PUNCT", "ADV", "ADJ", "VERB", "NOUN"]):
#     texts = [" ".join([token for token in doc if not token.is_stop and token.pos_ in pos]) for doc in docs]

#     return texts

def preprocess(raw_articles, labels=None):
    """
        Create pytorch dataloader from raw data
    """

    # https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_encode_plus.truncation

    encoded_batch = TOKENIZER.batch_encode_plus(raw_articles,
                                                add_special_tokens=False,
                                                padding = True,
                                                truncation = True,
                                                max_length = 128,
                                                return_attention_mask=True,
                                                return_tensors = 'pt')
        

    if labels:
        labels = torch.tensor(labels)
        return encoded_batch['input_ids'], encoded_batch['attention_mask'], labels
    return encoded_batch['input_ids'], encoded_batch['attention_mask']
##

# articles = preprocess_spacy(docs)
# print(TOKENIZER.convert_ids_to_tokens(preprocess(articles, labels=labels)[0][0]))

# %%
# Split train-validation
labels_train, articles_train = np.array(train_df[:, 1], dtype=int), train_df[:, 2]
labels_validation, articles_validation = np.array(test_df[:, 1], dtype=int), test_df[:, 2]

class_weights=class_weight.compute_class_weight('balanced',np.unique(labels_train),labels_train)
class_weights=torch.tensor(class_weights,dtype=torch.float)
 
print(class_weights) #([1.0000, 1.0000, 4.0000, 1.0000, 0.5714])


# %%
input_ids, attention_mask, labels_train = preprocess(articles_train, labels_train)
# Combine the training inputs into a TensorDataset
train_dataset = TensorDataset(
    input_ids,
    attention_mask,
    labels_train)
    

input_ids, attention_mask, labels_validation = preprocess(articles_validation, labels_validation)
# Combine the validation inputs into a TensorDataset
validation_dataset = TensorDataset(
    input_ids,
    attention_mask,
    labels_validation)


# %%
batch_size = 16

# Create the DataLoaders
train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size)

validation_dataloader = DataLoader(
            validation_dataset,
            sampler = SequentialSampler(validation_dataset),
            batch_size = batch_size)

# %%
model = CamembertForSequenceClassification.from_pretrained(
        'camembert-base',
        num_labels = 3)

# model.resize_token_embeddings(512)
# https://huggingface.co/transformers/v3.3.1/training.html
# https://towardsdatascience.com/visualize-bert-sequence-embeddings-an-unseen-way-1d6a351e4568

# %%
print(model.__dict__.keys())

# %%
def predict(articles, model=model):
    with torch.no_grad():
        model.eval()
        input_ids, attention_mask = preprocess(articles)
        output = model(input_ids, attention_mask=attention_mask)
        print(output[0])
        return torch.argmax(output[0], dim=1)
predict(articles_train[:10])

# Problem with the size of the embedding layer ???

# %%
def evaluate(articles, labels, metric='report'):
    predictions = predict(articles)
    if metric == 'report':
        return metrics.classification_report(labels, predictions, zero_division=0)
    elif metric == 'matrix':
        return metrics.confusion_matrix(labels, predictions)

# %%
def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    return str(datetime.timedelta(seconds=elapsed_rounded))

# %%
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # Learning Rate - Default is 5e-5
                  eps = 1e-8 # Adam Epsilon  - Default is 1e-8.
                )

# %%
SAVE_PATH = "sentences.model"

# Training loop
training_stats = []
                                                                                
# Measure the total training time for the whole run.
total_t0 = time.time()

epochs = 10

# Total number of training steps is [number of batches] x [number of epochs]
# (Note that this is not the same as the number of training samples)
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

device = torch.device("cpu")
criterion = torch.nn.CrossEntropyLoss(weight=class_weights,reduction='mean').to(device)

# This variable will evaluate the convergence on the training
consecutive_epochs_with_no_improve = 0

# Training
for epoch in range(0, epochs):
    
    print("")
    print(f'########## Epoch {epoch} / {epochs} ##########')
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    print('test time')

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode
    model.train()
    print('model.train')

    # For each batch of training data
    for step, batch in enumerate(train_dataloader):
        print(f'step : {step}')

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = time.time() - t0
            
            # Report progress
            print(f'  Batch {step}  of  {len(train_dataloader)}    Elapsed: {format_time(elapsed)}.')

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the 'device' using the 'to' method
        #
        # 'batch' contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: skills 
        input_id = batch[0].to(device)
        attention_mask = batch[1].to(device)
        label = batch[2].to(device)

        # Clear any previously calculated gradients before performing a backward pass
        model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch)
        # the loss (because we provided skills) and the "logits"--the model
        # outputs prior to activation
        loss, logits = model(input_id, 
                             token_type_ids=None, 
                             attention_mask=attention_mask, 
                             labels=label)[:2]
        
        

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. 'loss' is a Tensor containing a
        # single value; the '.item()' function just returns the Python value 
        # from the tensor
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients
        loss.backward()

        # Clip the norm of the gradients to 1.0
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches
    avg_train_loss = total_train_loss / len(train_dataloader)   

    if epoch > 0:
        if min([stat['Training Loss'] for stat in training_stats]) <= avg_train_loss:
            # i.e. If there is not improvement
            consecutive_epochs_with_no_improve += 1
        else:
            # If there is improvement
            consecutive_epochs_with_no_improve = 0
            print("Model saved!")
            torch.save(model.state_dict(), SAVE_PATH)
    
    # Measure how long this epoch took
    training_time = time.time() - t0

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))
    
    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Training Time': training_time,
        }
    )
    if consecutive_epochs_with_no_improve == 2:
        print("Stop training : The loss has not changed since 2 epochs!")
        break
        
torch.save(model.state_dict(), SAVE_PATH)

# %%
# Evaluation with the confusion matrix
confusion_matrix = evaluate(articles_validation, labels_validation, 'matrix')
report = evaluate(articles_validation, labels_validation, 'report')

print("PREDICT ON VALIDATION SET")
print(report)
print(confusion_matrix)
# seaborn.heatmap(confusion_matrix)
# precision    recall  f1-score   support

#            0       0.96      0.96      0.96       482
#            1       0.98      0.99      0.99      1322

#     accuracy                           0.98      1804
#    macro avg       0.97      0.97      0.97      1804
# weighted avg       0.98      0.98      0.98      1804

# %%
# confusion_matrix = evaluate(articles, labels, 'matrix')
# report = evaluate(articles, labels, 'report')
# print(report)
# seaborn.heatmap(confusion_matrix)

# %%
print("PREDICT ON Le réchauffement climatique est un complot des illuminatis")
print(predict(['Le réchauffement climatique est un complot des illuminatis']))

# %%



