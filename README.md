# Documentation

## Constitution of the dataset

### Scrapping & Extraction

> The first step was to scrap and extract the articles and then save them in json format in the “articles” directory.
> 

```json
# JSON format of the extracted articles

{
	"url": "",
	"title": "",
	"author": "",
	"date": "",
	"content": ""
}
```

- extractor_v1.py
    - utils.py
        
        
    - website_list.py

### Creation of the dataset and labellisation on TagTog

> We then selected 6,000 articles that would a priori give a balanced distribution of fake, true and biased articles. These articles are saved  in .txt format and are then uploaded to tagtog.
> 
- json_to_txt.py
    
    Save the content of the articles in .txt files in the "text_articles" folder.  the files are renamed "n.txt" where 0<n<6000. The information (title, date, author, url...) of the article n can be found in the dictionary "match_dictionary.pkl".
    

## Import of labelled articles from Tagtog

> Once the articles have been labelled, the label and annotations must be fetched from tagtog. Only at this stage did we decide to recover the paragraph structure of the articles which we had not initially collected. To do this we rescraped all the articles to recover the paragraphs, fetching for each article its label and its annotations thanks to the Tagtog API.
> 
- extract_paragraphs_ann.py
    
    Creation of json files corresponding to the articles annotated on tagtog. Two types of files are created, in the first type we keep the paragraph structure and the annotations ("json-annotations" folder). In the second type neither the paragraph structure nor the annotations are kept ("txt-labelled-articles" folder).
    
    - remove_useless.py
        
        Remove useless annotations from json files. Merges paragraphs.
        
    - paragraph_ann.py
        
        For a given article, creates a json file with the label, the structure into paragraphs and the annotations.
        
        - annotations.py
            - tagtog.py
        - paragraphs.py
    

**First type of Json file for labelled articles**

```jsx
{
"label": 3, // 1:True ; 2:Biased ; 3:Fake
"date": "2013",
"title":"",
"author":"",
"content": [ // List of Paragraphs
            {"content" : [ // List of entities
							{
	                "label": 3, // Fake annotation
	                "content": ""
	            },
	            {
	                "content": "" // Unannotated text between annotations
	            },
	            {
	                "label": 2, // Biased annotation
	                "content": ""
		         }]},

						{"content" : [ 
							{
	                "label": 3, 
	                "content": ""
	            },
	            {
	                "content": "" 
	            }]}
        ]
}
```

**Second type of Json file for labelled articles**

```jsx
{
"label": 3, // 1:True ; 2:Biased ; 3:Fake
"date": "2013",
"title":"",
"author":"",
"content":""
}
```

### 

### Preprocessing with Spacy

> The text is preprocessed with a spacy nlp pipeline.
> 
- spacy_save_docs.py
    
    generates a pickle file containing the Spacy pre-processing from a csv file containing the items and their labels
    

### Calculation of features

> The nlp preprocessing allows a number of features to be calculated
> 
- plot_feature.py
    
    plots the different distribution of each feature when applied to true, biased and false items
    
- features_to_pickle.py
    - creates a "features.p" file containing the features from the pre-processed items in "docs.p

### Formatting the dataset in a csv

> Data is stored in csv files and pickles. There are three types of data : whole articles, paragraphs and sentences. Thus the train dataset is not the same for each type of data. However test test and validation sets remain made up of whole articles as the test is done using the following prediction rule : if there is at least one fake paragraph or sentence then the article is predicted as fake too, if there is at least one biased paragraph or sentence and no fake one then the article is predicted as biased. Else the prediction is true.
> 

- articles_to_csv
    - creates a csv file containing all labelled items (found in the txt-labelled-articles folder) and their labels
    
- create_sets.py
    
    creates 3 pickle files (train, test, validation) for items, features and labels
    

### Classifiers

> Different classifiers are implemented (SVM, multinomial Naive Bayes classifier, different implementations of BERT...) and are grouped in the "classifier" folder.
> 

- BERT_torch.py
    
    BERT and a simple classifier taking as input simply the preprocessed articles in “docs.p”
    
- bert_with_features.py
    - Preprocessed articles go through BERT. Then selected features are concatenated to the CLS output. Finally the prediction is given by a dense layer.
- BERT_annotations.py
- BERT_paragraphs.py
    - Same as bert_with_features but training is done with paragraphs instead of whole articles.

- SVM_BOW.py

- LIME_NB.ipynb