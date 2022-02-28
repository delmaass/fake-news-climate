import os.path
from numpy import integer

from selenium import webdriver
from json import dumps
from pandas import isna

from utils.website_list import WEBSITE_LIST, INITIAL_WEBSITE_LIST
from utils.utils import DATABASE_PATH, SPECIAL_CHARACTERS, THRESHOLD_ON_CONTENT_LENGTH, TITLE_LENGTH, format_article_into_json, skip_cookie_popup, get_links, get_text_in_selected_element, get_date, standardize_date

driver = webdriver.Firefox()
# driver.maximize_window()

website_list = INITIAL_WEBSITE_LIST # Choose your website list

def extract(label, url, cookie_selector, link_selector, content_selector, title_selector, date_selector, author_selector, page_url_complement, number_of_pages, paginator_formula):
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

    #debug
    #number_of_pages = min(7, number_of_pages)

    for n in range(number_of_pages):

        full_link = url + page_url_complement.format(eval(paginator_formula))

        # Go to the URL
        driver.get(full_link)

        # Skip cookie pop-up on the first page
        if n==0:
            skip_cookie_popup(driver, cookie_selector)

        # Find all the article links on the page
        links = get_links(driver, link_selector)

        # DEBUG : 1 ARTICLE / WEBSITE
        #links = [links[0]]
        #print(links)

        #  Extract all articles from the page
        for link in links:
            driver.get(link)
            
            try :
                content = get_text_in_selected_element(driver, content_selector)

                if len(content) <= THRESHOLD_ON_CONTENT_LENGTH:
                    print(f"{link} : content too short")
                    continue

                # Get title
                title = get_text_in_selected_element(driver, title_selector)

                # Generate filename
                filename = label + "-" + title[:TITLE_LENGTH]
                for character in SPECIAL_CHARACTERS:
                    filename = filename.replace(character, '_')
                filename += ".json"

                # Check if this article has already been collected
                if os.path.isfile(DATABASE_PATH + filename):
                    print(filename + ": already collected")
                    continue

                author = get_text_in_selected_element(driver, author_selector)
                date = get_date(driver, date_selector)
                date = standardize_date(date)

                # Export article
                file = open(DATABASE_PATH + filename, "w")
                file.write(dumps(format_article_into_json(
                    url=link,
                    title=title,
                    author=author,
                    date=date,
                    content=content
                )))
                file.close()

            except Exception as e:
                print(repr(e))
                print(f"{label} : extraction failed")
                continue

def extract_website_list(website_list):
    for (label, url, cookie_selector, link_selector, content_selector, title_selector, date_selector, author_selector,page_url_complement, number_of_pages, paginator_formula) in website_list:
        extract(label, url, cookie_selector, link_selector, content_selector, title_selector, date_selector, author_selector, page_url_complement, number_of_pages, paginator_formula)
    driver.quit()

# Write logs
# file = open("logs.txt", "w")
# file.write(logs)
# file.close()

# print(f"{number_of_articles} articles")

# Utils
def extract_website(label):
    url = ""
    for website in WEBSITE_LIST:
        if website[0] == label:
            (label, url, cookie_selector, link_selector, content_selector, title_selector, date_selector, author_selector, page_url_complement, number_of_pages, paginator_formula) = website
    if not url:
        print("Incorrect label.")
        driver.quit()
    extract(label, url, cookie_selector, link_selector, content_selector, title_selector, date_selector, author_selector, page_url_complement, number_of_pages, paginator_formula)


def extract_from_website(label):
    website_list = []
    for (index, website) in enumerate(WEBSITE_LIST):
        if website[0] == label:
            website_list = WEBSITE_LIST[index:]
    if not website_list:
        print("Incorrect label.")
        driver.quit()
    extract_website_list(website_list)



extract_website_list(website_list)