from matplotlib.pyplot import get
import pandas as pd
import spacy
import os, json
from spacy import displacy
from pathlib import Path

# https://spacy.io/usage/linguistic-features

json_path = "json-annotations/FAKE/"
nlp = spacy.load("fr_core_news_md")

def get_fake_annotations():
    fake_annotations = []
    for _, filename in enumerate(os.listdir(json_path)):
        with open(os.path.join(json_path, filename), 'r') as file:
            if filename.startswith('.'):
                continue
            paragraphs = json.load(file)["content"]
            for paragraph in paragraphs:
                for entity in paragraph["content"]:
                    if type(entity) == dict and 'label' in entity.keys() :
                        if entity['label']==2:
                            fake_annotations.append(entity["content"])
    print(fake_annotations)
    return fake_annotations

# spacy is sentence ?
# Feature to plot ? dependency tree 

annotations = get_fake_annotations()
docs = [nlp(annotation) for annotation in annotations]

def save_images() :
    for index, doc in enumerate(docs[:20]):
        svg = displacy.render(doc, style="dep")
        file = open(os.path.join("./images", f'dep_{index}.svg'), "x", encoding="utf-8")
        file.write(svg)
        options = {"ents": ["PERSON", "ORG", "PRODUCT"],
           "colors": {"ORG": "yellow"}}
        #svg_ent = displacy.render(doc, style="ent", options = options)
        #file = open(os.path.join("./images", f'ent_{index}.svg'), "x", encoding="utf-8")
        #file.write(svg_ent)

def entities():
    entities = []
    for doc in docs:
        for ent in doc.ents:
            if ent.label_ == 'ORG' or ent.label_ == 'PER':
                entities.append((ent.text, ent.label_))
    unique_entities = list(set(entities))
    unique_entities_occurences = []
    for text, label in unique_entities :
        occurence = entities.count((text, label))
        unique_entities_occurences.append([text, label, occurence])
        unique_entities_occurences.sort(key=lambda x:-x[2])
    return unique_entities_occurences

def get_csv_from_entities():
    entities_list = entities()
    df = pd.DataFrame(entities_list, columns=['entity', 'entity type', 'occurence'])
    df.to_csv("images/entities.csv")
    print(df.head())

get_csv_from_entities()



