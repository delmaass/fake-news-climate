import json
import os
from annotations import get_annotations
from utils.tagtog import get_text_from_id

from spacy.training import offsets_to_biluo_tags
import spacy
import pandas as pd

ANN_JSON_PATH = "json-annotations/"
ANN_JSON_SUBFOLDER_PATH = ["TRUE/", "BIASED/", "FAKE/"]

def get_entities_from_id(article_id: int) -> dict:
    annotations = get_annotations(article_id)["annotations"]
    entities = []
    for ann in annotations:
        entity = [ann["start"], ann["start"] + len(ann["text"]), str(ann["label"])]
        entities.append(entity)
    return entities

def remove_useless_entities(content, entities):
    n_entities = len(entities)
    new_entities = []
    for idx, entity in enumerate(entities):
        # Remove useless entities (id 0)
        if entity[2] == "0":
            start, end = entity[:2]
            length = end - start
            # From content
            content = content[:start] + content[end:] 
            # From entities
            for i in range(idx, n_entities):
                entities[i][0] -= length
                entities[i][1] -= length
        else:
            new_entities.append((entity[0], entity[1], entity[2]))

    return content, new_entities

def get_content_entities_from_id(article_id):
    content = get_text_from_id(article_id)
    entities = get_entities_from_id(article_id)

    content, entities = remove_useless_entities(content, entities)

    return content, entities

def get_biluo(nlp, content, entities):
    doc = nlp(content)
    tags = offsets_to_biluo_tags(doc, entities)
    return tags

def get_ann_json_from_id(article_id):
    for subfolder in ANN_JSON_SUBFOLDER_PATH:
        file_path = ANN_JSON_PATH + subfolder + str(article_id) + ".json"
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                return json.load(file)
    print(file_path)
    raise Exception(f"No json file found for article #{article_id}")

def get_entities_per_par_from_json(json):
    entities_per_par = []

    content = json["content"]
    cursor = 0

    for par in content:
        entities = []
        subdivision = par["content"]
        if type(subdivision) == str:
            # There was initially no annotations in this paragraph
            cursor += len(subdivision)
            content_entities = (subdivision, entities)

        elif type(subdivision) == list:
            # There was initially at least one annotation in this paragraph
            subcontent = ""
            for ann in subdivision:
                ann_content = ann["content"]
                subcontent += ann_content

                if "label" in ann.keys():
                    # It is an annotation
                    entity = (cursor, cursor + len(ann_content), str(ann["label"]))
                    entities.append(entity)
                
                cursor += len(ann_content)
                content_entities = (subcontent, entities)
        else:
            # Wrong format
            raise Exception(f"Wrong format: {subdivision}")

        entities_per_par.append(content_entities)

    return entities_per_par

if __name__ == "__main__":
    nlp = spacy.load("fr_core_news_md")
    dataset = []

    for article_id in range(2800):
        try:
            j = get_ann_json_from_id(article_id)
            entities_per_par = get_entities_per_par_from_json(j)

            for par_ent in entities_per_par:
                content, entities = par_ent
                biluo = get_biluo(nlp, content, entities)
                dataset.append([content, ','.join(biluo)])
        except Exception as e:
            print(f"Skip {article_id}")
            print(repr(e))
            continue

    df = pd.DataFrame(dataset, columns=["text", "word_labels"])
    df.to_csv("word_labels_per_par.csv")

# if __name__ == "__main__":
#     article_id = 3096

#     nlp = spacy.load("fr_core_news_md")

#     content, new_entities = get_content_entities_from_id(article_id)
#     biluo = get_biluo(nlp, content, new_entities)

#     print(biluo)

# if __name__ == "__main__":
#     nlp = spacy.load("fr_core_news_md")

#     dataset = []

#     for article_id in range(2800):
#         try:
#             content, new_entities = get_content_entities_from_id(article_id)
#             biluo = get_biluo(nlp, content, new_entities)
#             dataset.append([content, ','.join(biluo)])
#         except Exception as e:
#             print(f"Skip {article_id}")
#             continue

#     df = pd.DataFrame(dataset, columns=["text", "word_labels"])
#     df.to_csv("word_labels_per_text.csv")