import os, json
import matplotlib.pyplot as plt
import numpy as np
from transformers import CamembertTokenizer, FlaubertTokenizer




camembert_tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
flaubert_tokenizer = flaubert_tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_base_cased')

def char_lengths(PATH):
    lengths = []
    for count, filename in enumerate(os.listdir(PATH)):
        with open(os.path.join(PATH, filename), 'r') as file:
            if filename.startswith('.'):
                continue
            paragraphs = json.load(file)["content"]
            for paragraph in paragraphs :
                text = ''
                for entity in paragraph["content"] :
                    if type(entity)==str :
                        text += entity
                    else :
                        text += entity["content"]
                lengths.append(len(text))
    return np.array(lengths)



def flaubert_token_lengths(PATH):
    lengths = []
    for count, filename in enumerate(os.listdir(PATH)):
        with open(os.path.join(PATH, filename), 'r') as file:
            if filename.startswith('.'):
                continue
            paragraphs = json.load(file)["content"]
            for paragraph in paragraphs :
                text = ''
                for entity in paragraph["content"] :
                    if type(entity)==str :
                        text += entity
                    else :
                        text += entity["content"]
                tokens = flaubert_tokenizer.encode(text, truncation=True, max_length = 512)
                lengths.append(len(tokens))
    return np.array(lengths)

def camembert_token_lengths(PATH):
    lengths = []
    for count, filename in enumerate(os.listdir(PATH)):
        with open(os.path.join(PATH, filename), 'r') as file:
            if filename.startswith('.'):
                continue
            paragraphs = json.load(file)["content"]
            for paragraph in paragraphs :
                text = ''
                for entity in paragraph["content"] :
                    if type(entity)==str :
                        text += entity
                    else :
                        text += entity["content"]
                tokens = camembert_tokenizer.encode(text, truncation=True, max_length = 512)
                lengths.append(len(tokens))
    return np.array(lengths)



if __name__ == '__main__':
    fig, axs = plt.subplots(3, 2)
    true_path = 'json-annotations/TRUE/'
    fake_path = 'json-annotations/FAKE/'
    biased_path = 'json-annotations/BIASED/'
    true_char_lengths=char_lengths(true_path)
    true_flau_lengths=flaubert_token_lengths(true_path)
    true_cam_lengths=camembert_token_lengths(true_path)
    biased_char_lengths=char_lengths(biased_path)
    biased_flau_lengths=flaubert_token_lengths(biased_path)
    biased_cam_lengths=camembert_token_lengths(biased_path)
    fake_char_lengths=char_lengths(fake_path)
    fake_flau_lengths=flaubert_token_lengths(fake_path)
    fake_cam_lengths=camembert_token_lengths(fake_path)
    axs[0, 0].hist(true_char_lengths, bins = np.arange(5, 1300, 10), color = 'green')
    axs[0, 0].set_title('Number of characters')
    axs[0, 0].set(ylabel='TRUE')
    axs[0, 0].axvline(true_char_lengths.mean(), color='k', linestyle='dashed', linewidth=1)
    axs[0, 1].hist(true_cam_lengths, bins = np.arange(5, 550), color = 'green')
    axs[0, 1].set_title('Number of camembert tokens')
    axs[0, 1].axvline(true_cam_lengths.mean(), color='k', linestyle='dashed', linewidth=1)
    axs[1, 0].hist(biased_char_lengths, bins = np.arange(5, 1300, 10), color = 'pink')
    axs[1, 0].set(ylabel='BIASED')
    axs[1, 0].axvline(biased_char_lengths.mean(), color='k', linestyle='dashed', linewidth=1)
    axs[1, 1].hist(biased_cam_lengths, bins = np.arange(5, 550), color = 'pink')
    axs[1, 1].axvline(biased_cam_lengths.mean(), color='k', linestyle='dashed', linewidth=1)
    axs[2, 0].hist(fake_char_lengths, bins = np.arange(5, 1300, 10), color = 'red')
    axs[2, 0].set(ylabel='FAKE')
    axs[2, 0].axvline(fake_char_lengths.mean(), color='k', linestyle='dashed', linewidth=1)
    axs[2, 1].hist(fake_cam_lengths, bins = np.arange(5, 550), color = 'red')
    axs[2, 1].axvline(fake_cam_lengths.mean(), color='k', linestyle='dashed', linewidth=1)
    fig.suptitle('Paragraphs lengths', fontsize = 16)
    plt.show()
