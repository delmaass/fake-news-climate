import json
from paragraph_ann import generate_paragraphs_ann, visualize
from remove_useless import remove_useless, fusion
from selenium import webdriver

json_path = "json-annotations/"
text_path = 'txt-labelled-articles/'
true_path = 'TRUE/'
fake_path = 'FAKE/'
biased_path = 'BIASED/'

def extract_paragraphs_ann(driver, article_id):
    paragraphs_ann = generate_paragraphs_ann(driver, article_id)
    paragraphs = remove_useless(paragraphs_ann)
    text_paragraphs = fusion(paragraphs)
    label = paragraphs['label']
    if label == 1 :
        path = json_path + true_path + str(article_id) + ".json"
        print(f"> Saving in {path}")
        with open(path, "w") as f:
            f.write(json.dumps(paragraphs))
        path = text_path + true_path + str(article_id) + ".json"
        print(f"> Saving in {path}")
        with open(path, "w") as f:
            f.write(json.dumps(text_paragraphs))
    elif label == 2 :
        path = json_path + biased_path + str(article_id) + ".json"
        print(f"> Saving in {path}")
        with open(path, "w") as f:
            f.write(json.dumps(paragraphs))
        path = text_path + biased_path + str(article_id) + ".json"
        print(f"> Saving in {path}")
        with open(path, "w") as f:
            f.write(json.dumps(text_paragraphs))
    elif label == 3 :
        path = json_path + fake_path + str(article_id) + ".json"
        print(f"> Saving in {path}")
        with open(path, "w") as f:
            f.write(json.dumps(paragraphs))
        path = text_path + fake_path + str(article_id) + ".json"
        print(f"> Saving in {path}")
        with open(path, "w") as f:
            f.write(json.dumps(text_paragraphs))


if __name__ == "__main__":
    driver = webdriver.Firefox()

    for article_id in range(1365, 3000):
        try:
            extract_paragraphs_ann(driver, article_id)
        except Exception as e:
            continue

    driver.quit()