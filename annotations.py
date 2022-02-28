from utils.tagtog import get_ann_from_article_id, get_ann_json, get_ann_legend, get_tagtog_id, get_text

DOCUMENT_LABEL_DICTIONARY = {
    "useless": 0,
    "true": 1,
    "biased": 2,
    "fake": 3
}

ANN_LABEL_DICTIONARY = {
    "useless": 0,
    "correct": 1,
    "biased": 2,
    "fake": 3
}

def get_label_from_ann(ann: dict) -> int:
    legend = get_ann_legend()

    label = ann["metas"][legend["label"]]["value"]

    if label not in DOCUMENT_LABEL_DICTIONARY.keys():
        raise Exception(f"This document has an unknown label '{label}'. Please update DOCUMENT_LABEL_DICTIONARY.")

    return DOCUMENT_LABEL_DICTIONARY[label]

def get_annotations_from_ann(ann: dict) -> list:
    annotations = []

    legend = get_ann_legend()

    entities = ann["entities"]
    for entity in entities:
        # Get the entity label
        class_id = entity["classId"]
        label = -1
        for key in ANN_LABEL_DICTIONARY.keys():
            if legend[key] == class_id:
                label = ANN_LABEL_DICTIONARY[key]
        if label == -1:
            print(f"Unknown label class found in the annotations: {class_id}. Skipping.")
            continue
        
        # Get the 'start' and 'text' attributes
        start = entity["offsets"][0]["start"]
        text = ''.join([offset["text"] for offset in entity["offsets"]])

        # Concatenate and add into annotations
        annotation = {"label": label, "start": start, "text": text}
        annotations += [annotation]

    return annotations

def get_annotations(id: int) -> dict:
    ann = get_ann_from_article_id(id)
    # Get label
    label = get_label_from_ann(ann)
    annotations = get_annotations_from_ann(ann)
    
    return {"label": label, "annotations": annotations}

def get_text_annotations_from_id(id: int):
    tagtog_id = get_tagtog_id(id)

    content = get_text(tagtog_id)
    ann = get_ann_json(tagtog_id)
    label = get_label_from_ann(ann)
    annotations = get_annotations_from_ann(ann)
    
    return content, {"label": label, "annotations": annotations}

if __name__ == '__main__':
    article_id = 667

    annotations = get_annotations(article_id)

    print(f"Annotations: {annotations}")