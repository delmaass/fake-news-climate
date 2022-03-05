import os
import pandas as pd
import json

true_path = 'txt-labelled-articles/TRUE/'
biased_path = 'txt-labelled-articles/BIASED/'
fake_path = 'txt-labelled-articles/FAKE/'

def add_to_lists(path):
    articles = []
    labels = []
    ids = []
    for _, filename in enumerate(os.listdir(path)):
        if filename.startswith('.'):
                continue
        with open(os.path.join(path, filename), 'r', encoding="utf8") as file:
            json_file = json.load(file)
            label = json_file['label']
            article = json_file['content']
            articles.append(article)
            labels.append(label)
            ids.append(filename[:-5])
    return ids, articles, labels
        

if __name__== '__main__':

    true_ids, true_articles, true_labels = add_to_lists(true_path)
    biased_ids, biased_articles, biased_labels = add_to_lists(biased_path)
    fake_ids, fake_articles, fake_labels = add_to_lists(fake_path)
    
    articles = true_articles + biased_articles + fake_articles
    labels = true_labels + biased_labels + fake_labels
    ids = true_ids + biased_ids + fake_ids

    df = pd.DataFrame(zip(ids, labels, articles), columns=['article id', 'label', 'article'])
    df = df.sample(frac=1).reset_index(drop=True)    
    df.to_csv('text_dataset.csv')
    print(df.shape)
    print(df.head())


