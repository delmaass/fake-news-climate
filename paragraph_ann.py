from annotations import get_annotations
from paragraphs import extract_paragraphs, get_article_data, get_article_location, get_url
from selenium import webdriver
import json

def generate_paragraphs_ann(driver: webdriver.Firefox, id: int) -> dict:
    path = get_article_location(id)
    label = path.split('-')[0].split('/')[1].replace('_', ' ')
    data = get_article_data(path)
    url = get_url(data)
    author = data['author']
    date = data['date']
    title = data['title']
    
    paragraphs = extract_paragraphs(driver, label, url)
    annotations = get_annotations(id)

    document_label = annotations["label"]
    annotations = annotations["annotations"]

    paragraphs_ann = {"label": document_label, "date": date, "title": title, "author": author, "content": []}

    cursor = 0
    ann_cursor = 0
    ann_max_cursor = len(annotations)

    for paragraph in paragraphs:
        if paragraph == "C’est maintenant que tout se joue… " or paragraph == "📨 Ne ratez plus d'articles qui vous intéressent": # Specific case of Reporterre
            return paragraphs_ann

        paragraph_ann = {"content": []}
        subcontent = paragraph

        if ann_cursor == ann_max_cursor:
            paragraph_ann["content"] = paragraph

        while ann_cursor < ann_max_cursor:
            annotation = annotations[ann_cursor]
            length = len(subcontent)
            
            if not length:
                break

            start = annotation["start"]
            ann_text = annotation["text"]
            ann_label = annotation["label"]

            if cursor + length <= start: # <= or < ?
                cursor += length

                if not subcontent.isspace():
                    if len(paragraph_ann["content"]):
                        neutral_content = {"content": subcontent}
                        paragraph_ann["content"].append(neutral_content)
                    else:
                        paragraph_ann["content"] = paragraph

                break
            else:
                subcursor = start - cursor
                stop = subcursor + len(ann_text)

                if subcursor < 0:
                    print("> This annotation is inside an other one.")
                    print("> Skipping.")
                    ann_cursor += 1
                    continue
                
                # Annotation between two paragraphs
                if length <= stop: #TODO: bug 772
                    print(f"> Splitting annotation #{ann_cursor}")
                    stop = length

                    ann_split_len = len(ann_text) - (length - subcursor)

                    ann_split_len = subcursor + len(ann_text) - length

                    ann_split = {
                        "label": ann_label,
                        "start": cursor + stop,
                        "text": ann_text[-ann_split_len:]
                    }

                    annotations[ann_cursor] = ann_split

                    ann_text = ann_text[:length - subcursor]
                else:
                    ann_cursor += 1

                ann = subcontent[subcursor:stop]

                if ann.lower() == ann_text.lower():
                    neutral, subcontent = subcontent[:subcursor], subcontent[stop:]
                    cursor += stop

                    if len(neutral) and not neutral.isspace():
                        neutral_content = {"content": neutral}
                        paragraph_ann["content"].append(neutral_content)

                    ann_content = {"label": ann_label, "content": ann_text}

                    if not ann_text.isspace():
                        if not len(neutral) and (not len(subcontent) or subcontent.isspace()):
                            paragraph_ann = ann_content
                        else:
                            paragraph_ann["content"].append(ann_content)

                    if ann_cursor == ann_max_cursor and (len(subcontent) and not subcontent.isspace()):
                        neutral_content = {"content": subcontent}
                        paragraph_ann["content"].append(neutral_content)
                else:
                    print(f"> The text of the annotation #{ann_cursor} doesn't match with the paragraphs.")
                    try:
                        print(f"Text in the annotation: {ann_text}")
                        print(f"Text in the paragraph: {ann}")
                    except Exception as e:
                        print(repr(e))
                    print("> Skipping.")

                    if ann_cursor == ann_max_cursor and not len(paragraph_ann["content"]):
                        paragraph_ann["content"] = paragraph

        if len(paragraph_ann["content"]):
            paragraphs_ann["content"].append(paragraph_ann)

    return paragraphs_ann

def visualize(obj):
    j = json.dumps(obj, ensure_ascii=False).encode('utf8')
    try:
        print(j.decode())
    except Exception as e:
        print("> One or multiple characters can't be decoded.")
        print(j)

if __name__ == '__main__':
    driver = webdriver.Firefox()

    article_id = 53 # 772
    paragraphs_ann = generate_paragraphs_ann(driver, article_id)
    visualize(paragraphs_ann)

    driver.quit()