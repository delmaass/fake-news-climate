from remove_useless import remove_useless, fusion
from paragraph_ann import generate_paragraphs_ann
from selenium import webdriver
import pandas as pd


def get_text_dataset(n) :
    driver = webdriver.Firefox()
    articles = []
    labels = []
    authors = []
    titles = []
    dates = []
    for article_id in range(n):
        try :
            paragraphs_ann = generate_paragraphs_ann(driver, article_id)
            data = fusion(remove_useless(paragraphs_ann))
            if data['label']!=0:
                articles.append(data['content'])
                labels.append(data['label'])
                authors.append(data['author'])
                titles.append(data['title'])
                dates.append(data['date'])

            else :
                print('useless article')

        except :
            print(f'ERROR WITH ARTICLE NO. {article_id}')
    
    dataset = pd.DataFrame(list(zip(labels, dates, authors, titles, articles)),
               columns =['label', 'date', 'author', 'titles', 'article'])
    driver.quit()
    return dataset
    
df = get_text_dataset(1000)
print(df.head())

df.to_csv('text_dataset.csv')