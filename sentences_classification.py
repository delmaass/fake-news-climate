import pandas as pd
import spacy
import torch
from transformers import CamembertForSequenceClassification, CamembertTokenizer

from extract_annotated_sentences import get_sentences_from_doc
# from extract_annotated_sentences import get_topic_sentences

MODEL_PATH = "models/sentences/articles_sentences.model"

def preprocess(raw_articles, tokenizer, labels=None):
    """
        Create pytorch dataloader from raw data
    """

    # https://huggingface.co/docs/transformers/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.batch_encode_plus.truncation

    encoded_batch = tokenizer.batch_encode_plus(raw_articles,
                                                add_special_tokens=False,
                                                padding = True,
                                                truncation = True,
                                                max_length = 128,
                                                return_attention_mask=True,
                                                return_tensors = 'pt')
        

    if labels:
        labels = torch.tensor(labels)
        return encoded_batch['input_ids'].to(torch.int64), encoded_batch['attention_mask'].to(torch.int64), labels.to(torch.int64)
    return encoded_batch['input_ids'].to(torch.int64), encoded_batch['attention_mask'].to(torch.int64)

def predict(articles, model, tokenizer):
    with torch.no_grad():
        model.eval()
        input_ids, attention_mask = preprocess(articles, tokenizer)
        output = model(input_ids, attention_mask=attention_mask)
        return torch.argmax(output[0], dim=1).tolist()

def load_model_tokenizer():
    model = CamembertForSequenceClassification.from_pretrained(
        'camembert-base',
        num_labels = 3
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    tokenizer = CamembertTokenizer.from_pretrained(
        'camembert-base',
        do_lower_case=True
    )
    return model, tokenizer

if __name__ == "__main__":
    ##
    # article_ids = range(3100)
    ##

    model, tokenizer = load_model_tokenizer()

    nlp = spacy.load("fr_core_news_md", exclude=["parser"])
    nlp.enable_pipe("senter")

    SETS = ["train", "test", "validation"]

    for set in SETS:
        df = pd.read_csv(f"datasets/articles/{set}_text_dataset.csv")
        data = []

        for article_id, row in df.iterrows():
            content = str(row["article"])

            doc = nlp(content)
            sentences = get_sentences_from_doc(doc)

            predicted_labels = predict(sentences, model, tokenizer)

            for sentence_id, (sentence, label) in enumerate(zip(sentences, predicted_labels)):
                data.append((article_id, sentence_id, label, sentence))
                print(f'({article_id} - {sentence_id}) {label}: {sentence}')

        result_df = pd.DataFrame(data, columns=["article_id", "sentence_id", "label", "text"])
        result_df.to_csv(f"{set}_sentences_classification.csv")
        
