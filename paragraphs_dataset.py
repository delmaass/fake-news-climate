import os
import pandas as pd
import json

true_path = 'json-annotations/TRUE/'
biased_path = 'json-annotations/BIASED/'
fake_path = 'json-annotations/FAKE/'


def split_paragraphs(json_article):
    content = json_article["content"]
    paragraphs = []
    for paragraph in content :
        if type(paragraph["content"]) == str :
            paragraphs.append(paragraph["content"])
        else :
            text = ""
            for entity in paragraph["content"]:
                text += entity["content"]
            paragraphs.append(text)
    return paragraphs

def add_to_lists(path):
    paragraphs_data = []
    labels_data = []
    ids_data = []
    for _, filename in enumerate(os.listdir(path)):
        if filename.startswith('.'):
                continue
        with open(os.path.join(path, filename), 'r', encoding="utf8") as file:
            json_file = json.load(file)
            paragraphs = split_paragraphs(json_file)
            labels = [json_file['label']]*len(paragraphs)
            ids = [filename[:-5]]*len(paragraphs)
            labels_data+=labels
            paragraphs_data+=paragraphs
            ids_data+=ids

    return ids_data, paragraphs_data, labels_data

if __name__== '__main__':

    true_ids, true_pargraphs, true_labels = add_to_lists(true_path)
    biased_ids, biased_paragraphs, biased_labels = add_to_lists(biased_path)
    fake_ids, fake_paragraphs, fake_labels = add_to_lists(fake_path)
    
    paragraphs = true_pargraphs + biased_paragraphs + fake_paragraphs
    labels = true_labels + biased_labels + fake_labels
    ids = true_ids + biased_ids + fake_ids

    df = pd.DataFrame(zip(ids, labels, paragraphs), columns=['article id', 'label', 'paragraph'])
    df = df.sample(frac=1).reset_index(drop=True)    
    df.to_csv('paragraphs_dataset.csv')
    print(df.shape)
    print(df.head())     