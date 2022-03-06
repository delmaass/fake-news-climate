import os
import random
import pickle


true_path = 'json-annotations/TRUE/'
biased_path = 'json-annotations/BIASED/'
fake_path = 'json-annotations/FAKE/'


def get_ids(path):
    ids_list = []
    for _, filename in enumerate(os.listdir(path)):
        if filename.startswith('.'):
            continue
        ids_list.append(int(filename[:-5]))
    split_border_1 = int(len(ids_list)*0.7)
    split_border_2 = int(len(ids_list)*0.85)
    random.shuffle(ids_list)
    ids_train, ids_test, ids_validation = ids_list[:split_border_1], ids_list[split_border_1:split_border_2], ids_list[split_border_2:]
    return ids_train, ids_test, ids_validation

if __name__ == "__main__":
    true_train, true_test, true_val = get_ids(true_path)
    biased_train, biased_test, biased_val = get_ids(biased_path)
    fake_train, fake_test, fake_val = get_ids(fake_path)
    ids_train = true_train + biased_train + fake_train
    ids_test = true_test + biased_test + fake_test
    ids_val = true_val + biased_val + fake_val

    pickle.dump(ids_train, open("ids_train.p", "wb"))
    pickle.dump(ids_test, open("ids_test.p", "wb"))
    pickle.dump(ids_val, open("ids_val.p", "wb"))

    
        