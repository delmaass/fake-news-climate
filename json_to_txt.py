import os, json
import numpy as np
import pickle
from utils.topic_context import TOPIC_LEMMA, remove_accents

# Function to rename multiple files
def main():

    dictionary_of_matches = {}
    json_folder = "fake-news-excess"
    text_folder = "fake-news-articles"
    n = len(os.listdir(json_folder))
    numbering = np.arange(n) + 10000
    np.random.shuffle(numbering)

    for count, filename in enumerate(os.listdir(json_folder)):
        with open(os.path.join(json_folder, filename), encoding="utf8") as json_file:
            json_text = json.load(json_file)
            content = json_text['content']
            check_content = remove_accents(content)
            skip = True

            for lemma in TOPIC_LEMMA:
                if lemma in check_content:
                    skip = False
                    break

            if not skip:
                file = open(os.path.join(text_folder, f"{numbering[count]}.txt"), "w", encoding="utf8") 
                file.write(content) 
                file.close() 
                dictionary_of_matches[numbering[count]] = filename
    a_file = open("match_dictionary_fake.pkl", "wb")
    pickle.dump(dictionary_of_matches, a_file)
    a_file.close()
    a_file = open("match_dictionary_fake.pkl", "rb")
    output = pickle.load(a_file)
    print(output)

if __name__ == '__main__':
     
    main()