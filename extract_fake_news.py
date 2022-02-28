import json
import os
import pickle

import pandas as pd

def is_not_useless(content: str) -> bool:
    MUST_HAVE_WORDS = ["climatique", "climat", "réchauffement", "GIEC", "COP"]

    for word in MUST_HAVE_WORDS:
        if content.find(word) != -1:
            return True

    return False

def remove_useless_parts(content: str) -> str:
    USELESS_PARTS = ["Partagez ! Volti. ****** ", "Lire aussi : ", "l'Express", "EN VIDÉO >> ", "LIRE AUSSI >> ", "Contrepoints "]

    for part in USELESS_PARTS:
        content = content.replace(part, '')

    return content

if __name__ == "__main__":
    FAKE_PATH = "fake-news-excess/"
    fake_count = 0
    fake_dataset = []

    for filename in os.listdir(FAKE_PATH):
        if not filename.startswith("Wikistrike"):
            file_path = FAKE_PATH + filename
            if(os.path.isfile(file_path)):
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    label, content = 1, data["content"]
                    
                    if is_not_useless(content):
                        content = remove_useless_parts(content)
                        fake_dataset.append((label, content))

                        fake_count += 1

    fake_df = pd.DataFrame(fake_dataset, columns=["label", "article"])

    print(f'Fake: {fake_count}')
    print(fake_df.head())

    USUAL_PATH = "text_articles/"
    MATCH_DICT = pickle.load(open("match_dictionary.pkl", "rb"))
    TRUE_WEBSITES = ['L\'Express', 'Jean Marc Jancovici', 'Ouest France', 'Futura Planet', 'Carbone 4', 'Euronews']

    true_count = 0
    true_dataset = []

    for i in range(3100, 6050):
        initial_filename = MATCH_DICT[i]
        website = ' '.join(initial_filename.split('-')[0].split('_'))
        if website in TRUE_WEBSITES:
            filename = f'{i}.txt'
            file_path = USUAL_PATH + filename
            with open(file_path, 'r', encoding="utf-8") as file:
                label, content = 0, file.read()
                if is_not_useless(content):
                    content = remove_useless_parts(content)
                    true_dataset.append((label, content))

                    true_count += 1

    true_df = pd.DataFrame(true_dataset, columns=["label", "article"])

    print(f'True: {true_count}')
    print(true_df.head())

    INITIAL_CSV = pd.read_csv("text_dataset.csv")


    csv = INITIAL_CSV[INITIAL_CSV.label != 2]
    csv = csv.drop(columns=['Unnamed: 0'])
    csv.label = csv.label.map({1: 0, 3: 1})

    print(f'Initial dataset: {csv.label.value_counts()}')

    final_df = pd.concat([true_df, csv], axis=0, ignore_index=True)
    print(f'w/ True: {final_df.label.value_counts()}')

    fake_df = fake_df.sample(frac=1).reset_index(drop=True)
    fake_to_add = final_df.label.value_counts()[0] - csv.label.value_counts()[1]
    final_df = pd.concat([final_df, fake_df[:fake_to_add]], axis=0, ignore_index=True)
    print(f'w/ Fake: {final_df.label.value_counts()}')

    final_df = final_df.sample(frac=1).reset_index(drop=True)
    final_df.to_csv("2040_true_fake.csv")