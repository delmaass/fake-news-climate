from pandas import isna
from selenium import webdriver
from json import load
from utils.tagtog import get_ann_from_article_id, get_ann_legend
from utils.utils import DATABASE_PATH, has_bad_prefix, has_very_bad_prefix, skip_cookie_popup
from utils.website_list import WEBSITE_LIST
import pickle
import numpy as np

with open("datasets/match_dictionary.pkl", "rb") as file:
    MATCH_DICTIONARY = pickle.load(file)

# Utils
def get_article_location(id: int) -> str:
    return DATABASE_PATH + MATCH_DICTIONARY[id]

def get_article_data(path: str) -> dict:
    with open(path) as file:
        return load(file)

def get_url(data: dict) -> str:
    return data["url"]

def go_to_url(driver: webdriver.Firefox, url: str):
    driver.get(url)

def get_selectors(label: str):
    print(label)
    (cookie_selector, link_selector, content_selector, title_selector, date_selector, author_selector, page_url_complement, number_of_pages, paginator_formula) = list(filter(lambda line: line[0] == label, WEBSITE_LIST))[0][2:]

    # Replace by default values if empty string
    cookie_selector = "x" if isna(cookie_selector) else cookie_selector
    link_selector = "article a" if isna(link_selector) else link_selector
    content_selector = "article" if isna(content_selector) else content_selector
    title_selector = "h1" if isna(title_selector) else title_selector
    date_selector = "x" if isna(date_selector) else date_selector
    author_selector = "x" if isna(author_selector) else author_selector
    page_url_complement = "" if isna(page_url_complement) else page_url_complement
    number_of_pages = 1 if isna(number_of_pages) else int(number_of_pages)
    paginator_formula = "n" if isna(paginator_formula) else paginator_formula

    return cookie_selector, link_selector, content_selector, title_selector, date_selector, author_selector, page_url_complement, number_of_pages, paginator_formula

# Paragraphs
def collect_paragraphs(driver: webdriver.Firefox, content_selector: str) -> list:
    content_elements = driver.find_elements_by_css_selector(content_selector)
    paragraphs = []
    
    for i, content_element in enumerate(content_elements):
        text = content_element.text


        if has_very_bad_prefix(text) :
            print("Very bad prefix")
            break

        if not has_bad_prefix(text) or (i>2 and (i < len(content_elements)-5)):
            if len(content_elements) > 2:
                paragraphs += [text + " "]
            else:
                lines = text.split('\n')
                if len(lines):
                    lines = [line + "\n" for line in lines]
                    if not text.endswith('\n'):
                        lines[-1] = lines[-1][:-1]
                    lines[-1] += " "
                    paragraphs += lines
                else:
                    paragraphs += [text + " "]

    return paragraphs

def extract_paragraphs(driver: webdriver.Firefox, label: str, url: str) -> list:
    go_to_url(driver, url)
    cookie_selector, _, content_selector, _, _, _, _, _, _ = get_selectors(label)
    
    # Skip cookie pop-up on the first page
    skip_cookie_popup(driver, cookie_selector)
        
    try :
        # Collect paragraphs according to the 'content_selector'
        paragraphs = collect_paragraphs(driver, content_selector)
        return paragraphs
    except Exception as e:
        print(repr(e))
        print(f"({label}) {url} : Failure")
        return None

def visualize(obj):
    try:
        print(obj)
    except Exception as e:
        print("> One or multiple characters can't be decoded.")
        obj = [o.encode("utf-8") for o in obj]
        print(obj)

if __name__ == '__main__':
    driver = webdriver.Firefox()

    article_id = 59

    path = get_article_location(article_id)
    label = path.split('-')[0].split('/')[1].replace('_', ' ')
    data = get_article_data(path)
    url = get_url(data)

    paragraphs = extract_paragraphs(driver, label, url)
    visualize(paragraphs)
    
    driver.quit()