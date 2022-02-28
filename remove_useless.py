from paragraph_ann import generate_paragraphs_ann
from selenium import webdriver

def remove_useless(paragraph_ann):
    content = paragraph_ann["content"]
    paragraphs_to_delete = []
    entities_to_delete = []

    for index, paragraph in enumerate(content) :
        if 'label' in paragraph.keys() and paragraph["label"]==0:
            paragraphs_to_delete.append(index)

        else :
            if type(paragraph["content"])==list:
              for index_entity, entity_content in enumerate(paragraph["content"]):
                if type(entity_content) == dict and 'label' in entity_content.keys() and entity_content["label"]==0:
                      entities_to_delete.append((index, index_entity))

    for i in range(-1, -len(entities_to_delete)-1, -1) :
      index, index_entity = entities_to_delete[i]
      del paragraph_ann['content'][index]['content'][index_entity]

    for index in reversed(paragraphs_to_delete):
      del paragraph_ann['content'][index]

    return paragraph_ann


def fusion(paragraph_ann):
  label = paragraph_ann['label']
  author = paragraph_ann['author']
  title = paragraph_ann['title']
  date = paragraph_ann['date']
  content = ""
  for paragraph in paragraph_ann['content']:
    if type(paragraph['content']) == str :
      content += paragraph['content']
    elif type(paragraph['content']) == list :
      for entity in paragraph['content']:
        content += entity['content']
  json = {
    'label' : label, 
    'date': date,
    'title': title,
    'author': author,
    'content' : content,
  };
  return json




article_id = 11
driver = webdriver.Firefox()
paragraphs_ann = generate_paragraphs_ann(driver, article_id)
print(fusion(remove_useless(paragraphs_ann)))
driver.quit()