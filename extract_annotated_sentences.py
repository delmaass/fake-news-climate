import pandas as pd
import spacy
from spacy.tokens import Doc
from spacy.lang.fr import French
from annotations import get_text_annotations_from_id
from topic_context import remove_accents, TOPIC_LEMMA

def get_sentences_from_doc(doc: Doc):
    return [sentence.text for sentence in doc.sents]

# get_clauses_from_doc ? cf. topic_context.py

def get_annotated_sentences(article_id: int, sentences: list[str], entities: list) -> list:
    LABELS = [2, 3]
    entities = list(filter(lambda entity: entity["label"] in LABELS, entities))
    # print(entities)

    cursor = 0
    cursor_entities = 0
    cursor_sentences = 0
    n_entities = len(entities)
    n_sentences = len(sentences)
    annotated_sentences = []

    if n_entities:
        while cursor_sentences < n_sentences:
            sentence = sentences[cursor_sentences]
            entity = entities[cursor_entities]

            start = entity["start"]
            end_sentence = cursor + len(sentence)

            # print(cursor, start, end_sentence)
            if cursor <= start < end_sentence:
                # Annotated sentence !
                label = entity["label"]
                annotated_sentences.append((article_id, label, sentence))

                # But we have to consider that the annotation may goes on an other sentence
                end = start + len(entity["text"])
                # print(end)
                if end > end_sentence:
                    # print(entity["text"])
                    entity["start"] = end_sentence + 1
                    entity["text"] = entity["text"][-(end-end_sentence)+1:]
                    # print(entity["text"])
                else:
                    cursor_entities += 1

                cursor_sentences += 1
                cursor = end_sentence + 1 # The space between 2 par
            elif start < cursor:
                cursor_entities += 1
            else:
                cursor_sentences += 1
                cursor = end_sentence + 1

            if cursor_entities >= n_entities:
                break
    
    return annotated_sentences

def get_topic_sentences(article_id: int, sentences: list[str], label=1) -> list:
    # We look for sentences that :
    # 1. Deals with climate change
    # 2. Are not quotations and not questions and not 1st person
    QUOTATION_MARKS = ['"', '«', '»', '?', 'je']
    annotated_sentences = []

    for sentence in sentences:
        annoting = False
        check_sentence = remove_accents(sentence).lower()

        for lemma in TOPIC_LEMMA:
            lemma = lemma.lower()

            if lemma in check_sentence:
                annoting = True
                break

        for mark in QUOTATION_MARKS:
            if mark in check_sentence:
                annoting = False
                break

        if annoting:
            annotated_sentences.append((article_id, label, sentence))

    return annotated_sentences

# nlp = French()  # just the language with no pipeline
# nlp.add_pipe("sentencizer")
# OR 
# nlp = spacy.load("fr_core_news_md")

if __name__ == "__main__":
    nlp = spacy.load("fr_core_news_md", exclude=["parser"])
    nlp.enable_pipe("senter")

    dataset = []

    N_ARTICLES = 3100
    for article_id in range(N_ARTICLES):
        try:
            content, annotations = get_text_annotations_from_id(article_id)
            label, entities = annotations["label"], annotations["annotations"]
            # print(f"Label: {label}")

            if label == 0: # Useless
                print(f"Useless #{article_id}")
                continue

            doc = nlp(content)
            sentences = get_sentences_from_doc(doc)
            # print(sentences)

            if label == 1:
                annotated_sentences = get_topic_sentences(article_id, sentences)
            else:
                annotated_sentences = get_annotated_sentences(article_id, sentences, entities)

            # print(annotated_sentences)
            dataset += annotated_sentences
        except Exception as e:
            print(f"Skipping #{article_id}")
        
    df = pd.DataFrame(dataset, columns=["article_id", "label", "text"])
    df = df.sample(frac=1).reset_index(drop=True)
    
    df.to_csv("articles_annotated_sentences.csv")
        
        
