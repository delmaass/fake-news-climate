import os
import random
import pickle
import pandas as pd


file_to_read = open("match_dictionary.pkl", "rb")
match_dictionary = pickle.load(file_to_read)

true_path = 'json-annotations/TRUE/'
biased_path = 'json-annotations/BIASED/'
fake_path = 'json-annotations/FAKE/'

true_websites = ['Ouest_France', 'Carbone_4', 'Jean_Marc_Jancovici', 'L\'Express', 'Euronews', 'Futura_Planet', 'The_Conversation']
mixed_websites = ['CPEPESC',  'Les_Crises', 'Bon_Pote', 'Reporterre', 'GreenPeace',  'Time_for_the_Planet', 'Wikipedia', 'OXFAM_France', 'Usbek_&_Rica']
fake_websites = ['Les_moutons_enragés','Egalité_&_Réconciliation', 'Arrêt_Sur_Info', 'SytiNet', 'Réseau_Voltaire', 'Association_des_climatorealistes', 'Contrepoints', 'Réseau_International', 'Wikistrike']

test_websites = []
train_websites = []

def get_stats(path):
    path_total_articles = 0
    from_true = 0
    from_mixed = 0
    from_fake = 0
    for _, filename in enumerate(os.listdir(path)):
        if filename.startswith('.'):
            continue
        path_total_articles += 1
        file_id = int(filename[:-5])
        for website in true_websites:
            if str(match_dictionary[file_id]).startswith(website):
                from_true += 1
                print(match_dictionary[file_id])
                print(file_id)
        for website in mixed_websites:
            if str(match_dictionary[file_id]).startswith(website):
                from_mixed += 1
        for website in fake_websites:
            if str(match_dictionary[file_id]).startswith(website):
                from_fake += 1

    return path_total_articles, from_fake, from_mixed, from_true


if __name__ == "__main__":
    #print(get_stats(true_path))
    #print(get_stats(biased_path))
    print(get_stats(fake_path))

