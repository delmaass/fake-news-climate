from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re

DATABASE_PATH = "articles/"
SPECIAL_CHARACTERS = [
    ' ', '/', '<', '>', ':', '"', '\\', '|', '?', '*'
]
THRESHOLD_ON_CONTENT_LENGTH = 1000
TITLE_LENGTH = 40
BAD_PREFIXES = ['Source', 'Tradu', 'Ajout', 'Lire', 'Commentaire', 'Partage', 'http', 'www', 'Inscri', 'Lien', 'Article', 'Direction', 'Extrait', 'Voir', 'Cet article', 'Abonnez']
VERY_BAD_PREFIXES = ['Note', 'Partager', "S'abonner", 'Abonnez', '\ud83d', 'Voir aussi :']

# General functions
def format_article_into_json(url, title, author, date, content):
    return {
        "url": url,
        "title": title,
        "author": author,
        "date": date,
        "content": content
    }


# Extraction functions
def skip_cookie_popup(driver, selector):
    if selector != "x":
        try:
            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR,selector))).click()
        except Exception as e:
            print(repr(e))
            print("cookie bug")
            return


def get_links(driver, css_selector):
    link_elements = driver.find_elements_by_css_selector(css_selector)
    links = []
    for link_element in link_elements:
        link = link_element.get_attribute("href")
        if link not in links : # remove duplicates
            links += [link]
    return links

def has_bad_prefix(text) -> bool:
    for prefix in BAD_PREFIXES:
        if bool(re.match(prefix, text, re.I)):
            return True

    return False

def has_very_bad_prefix(text) -> bool:
    for prefix in VERY_BAD_PREFIXES:
        if bool(re.match(prefix, text, re.I)):
            return True

    return False

def get_text_in_selected_element(driver, selector):
    if selector == "x":
        return ""
    try:
        content_elements = driver.find_elements_by_css_selector(selector)
        content = ""

        for i, content_element in enumerate(content_elements):
            text = content_element.text

            if has_very_bad_prefix(text) :
                print("Very bad prefix")
                break

            if not has_bad_prefix(text) or (i>2 and (i < len(content_elements)-5)):
                content += text
                content += " "

        return content
    except Exception as e:
        print(repr(e))
        return ""

def get_date(driver, selector):
    try:
        element = driver.find_element_by_css_selector(selector)
        datetime = element.get_attribute("datetime")
        if datetime is not None:
            return datetime
        return get_text_in_selected_element(driver, selector)
    except Exception as e:
        print(repr(e))
        return ""

def standardize_date(date) :
    consecutive_numbers = 0
    year = ""
    for c in date :
        if(ord(c) >= 48 and ord(c) <= 57) :
            consecutive_numbers+=1
            year += c
        else :
            consecutive_numbers=0
            year = ""
        if consecutive_numbers ==  4:
            return year
    return ""
