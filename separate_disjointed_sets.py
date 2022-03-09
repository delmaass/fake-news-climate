import os
import random
import pickle
import pandas as pd


file_to_read = open("match_dictionary.pkl", "rb")
match_dictionary = pickle.load(file_to_read)

true_path = 'json-annotations/TRUE/'
biased_path = 'json-annotations/BIASED/'
fake_path = 'json-annotations/FAKE/'

test_websites = ['Association_des_climatorealistes', 'Reporterre', 'Contrepoints', 'Usbek_&_Rica', 'Ouest_France', 'Arrêt_Sur_Info', 'Wikipedia', 'Carbone_4', 'CPEPESC']
train_websites = ['Wikistrike', 'L\'Express', 'Futura_Planet', 'The_Conversation', 'GreenPeace', 'Réseau_International', 'Les_moutons_enragés', 'Les_Crises', 'Jean_Marc_Jancovici', 'Egalité_&_Réconciliation', 'Bon_Pote', 'Réseau_Voltaire', 'SytiNet', 'Time_for_the_Planet']



def get_ids(path):
    ids_train_list = []
    ids_test_list = []
    for _, filename in enumerate(os.listdir(path)):
        if filename.startswith('.'):
            continue
        file_id = int(filename[:-5])
        for website in train_websites:
            if str(match_dictionary[file_id]).startswith(website):
                ids_train_list.append(file_id)
        for website in test_websites:
            if str(match_dictionary[file_id]).startswith(website):
                ids_test_list.append(file_id)

    return ids_train_list, ids_test_list

if __name__ == "__main__":
    true_train, true_test = get_ids(true_path)
    biased_train, biased_test = get_ids(biased_path)
    fake_train, fake_test = get_ids(fake_path)
    ids_train = true_train + biased_train + fake_train
    ids_test = true_test + biased_test + fake_test
    df = pd.read_csv('text_dataset.csv')
    train_df = df.loc[df['article id'].isin(ids_train)]
    test_df = df.loc[df['article id'].isin(ids_test)]
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)
    train_df.to_csv('train_disjoint.csv')
    test_df.to_csv('test_disjoint.csv')