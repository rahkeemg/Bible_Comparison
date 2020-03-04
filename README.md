# Abstract

The Bible is the best selling book of all times and continues to be a source for many pointing towards God. Originally written in Hebrew, Aramaic, and Greek, the Bible has been translated into over 600+ languages, with each respective language having its own respective variations and verisons.  

In this project, we will be examinign a few English Translations of the Bible and using some natural language procesing (NLP) tools in attempt to see how some of these versions are similar and how they differ.

The goal of this project is not to declare which versions are better or worse than another, but simply to see how these versions relate to one another, using Python and some machine learning techniques to quantify and make comparisons.

___________________________

# Project Overview

In order to compare the different Bibles versions, we will first need a source to pull the Bibles from.  For this project, we will utilize the following online APIs to pull Bible text:

* Biblia
* GetBible

Below is a diagram on what this project will look like:

![Project_Overview](./assets/resources/images/Capstone_Project_Overview.png "Overview")

_________________________________________________

### Import Files


```python
import requests
import pandas as pd
import numpy as np
import json
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
import cv2
import nltk
from mpl_toolkits.axes_grid1 import ImageGrid
from plotly.subplots import make_subplots
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from keras import models
from keras.layers import Dense, Activation
from gensim.models import Word2Vec
from PIL import Image
from API import My_API
import text_handling as th

%load_ext autoreload
%autoreload 2
%matplotlib inline
```

    Using TensorFlow backend.
    C:\Users\Rahkeem\Anaconda3\envs\learn-env\lib\site-packages\gensim\utils.py:1197: UserWarning:
    
    detected Windows; aliasing chunkize to chunkize_serial
    
    

# Accessing APIs for Bible text

To store the different versions of the Bible, we will use a dictionary as the main structure to hold the content of each, called `versions`


```python
versions = {}
```

In this section, the most care had to be taken into consideration when pulling information from each API.  In an attempt to access the text as easily as possible, I am using a custom API to accesss each site. The queries are inserted and built by hand using the respective documentation at each site.  

#### Cannonical books

The API _Biblia API_ is a very useful site that is well documented and also has services built in to get the contents fo each specific bible.  To make the text retrieval process more seamless, the list of books within standard canonical bibles are retrieved, then stored for later use within different queries.


```python
## Pull the canonical books from Website
my_api = My_API(url="https://api.biblia.com/v1/bible/", key='fd37d8f28e95d3be8cb4fbc37e15e18e')
query = 'contents/KJV?'
resp = my_api.run_query(query)

## Strip out the books and remvoe whitespace with '_' 
# canonical_books = [book['passage'].replace(" ","_").strip() for book in resp['books']]
canonical_books = [book['passage'].strip() for book in resp['books']]
```


```python
print(canonical_books)
```

### Custom functions for Bible text retrieval and formatting

Text retrieval was the most difficult part of the project. Each API differs significantly and the format of the text returned made this even more difficult. To take care of these differences, custom functions were created and stored in the helper filed, _Text_Handling.py_

Format of downloaded text

To simplify the retrieval and storage of Bible text across different platforms, the text is stored as fully formatted verse as seen in the example below.

```
{"KJV":
   {
    "Genesis 1:1": "In the beginning God created heaven, and earth.",
    "Genesis 1:2": "And the earth was void and empty, and darkness was upon the face of the deep; and the spirit of God moved over the waters.",
    "Genesis 1:3": "And God said: Be light made. And light was made.",
    "Genesis 1:4": "And God saw the light that it was good; and he divided the light from the darkness.",
    ...
    "Revelation 22:21": "The grace of our Lord Jesus Christ be with you all. Amen."
    }
}
```

### _Biblia API_

This site is very useful and hosts a strong amount of services for pulling Bible text from the website.
Finding the right query to use was the most difficiult part of using this query.  

As a result, the decision was made to use the following versions below, and offset the remaining versions to another API:

Bible Version |	Version ID  
:---:     |:----:  |
1890 Darby Bible |	DARBY  
The Emphasized Bible |	EMPHBBL 
King James Version |	KJV1900  
The Lexham English Bible |	LEB


```python
biblia_available_versions = ['KJV1900','LEB','EMPHBBL','DARBY']
```

To get the content from a specific bible, the `content` service of the API has to be used.  Selecting different services is pretty simple.  To do so, you enter the name of the service as a path, within a directory as seen below.  

Pull the text for each book from website and store in dictionary for later use


```python
my_api = My_API(url="https://api.biblia.com/v1/bible/", key='fd37d8f28e95d3be8cb4fbc37e15e18e')

biblia_versions = {}

for v in biblia_available_versions:
    print(v)
    ver = {}
    for book in canonical_books:

        query = f'content/{v}.txt?passage={book}&style=oneVersePerLineFullReference'
        resp = my_api.run_query(query)

        book_dict = th.parse_formatted_verse_ref(resp)
        ver.update(book_dict)
        
    versions[v] = ver
```


```python
versions.keys()
```

### _GetBible.net_

This site was the easiest to use and gave the information in the ideal format.
Each verse was in its own respective JSON array, requiriing less effort to retrieve the information.

Below are the English BIble Versions available at this website:

Bible Version |	Version ID  
:---:     |:----:  
American Standard Version |	ASV 
Authorized Version |	KJV  
The Lexham English Bible |	LEB 
Young’s Literal Translation |	YLT
BasicEnglish Bible  | BASICENGLISH
Douary Rheims Bible | DOUAYRHEIMS
Webster's Bible  | WB
World English Bible | WEB



```python
getbible_eng_versions = ['KJV','AKJV','ASV','BASICENGLISH','DOUAYRHEIMS', 'WB','WEB','YLT']
```


```python
my_api = My_API(url="https://getbible.net/")

for v in getbible_eng_versions:
    v_dict = {}
    print(v)
    for book in canonical_books:
        resp = my_api.run_query(f'json?passage={book}&v={v.lower()}')
        resp_dict = json.loads(resp[1:-2])
        v_dict.update(th.parse_book(resp_dict))
    versions[v] = v_dict
```


```python
versions.keys()
```

### Save data to file

Here, we are saving the dictionary as a json object to a file, for easier access later on


```python
for key in versions.keys():
    key_dict = {key: versions[key]}
    with open(f"./data/{key}.json", "w") as f:
        js = json.dump(key_dict, fp=f, indent=4, separators=(',', ': '))
```

### Pull data from saved file

Pulling files from `/data` directory


```python
from os import listdir
from os.path import isfile, join

mypath = './data'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
```


```python
onlyfiles
```




    ['AKJV.json',
     'ASV.json',
     'BASICENGLISH.json',
     'DARBY.json',
     'DOUAYRHEIMS.json',
     'EMPHBBL.json',
     'KJV.json',
     'KJV1900.json',
     'LEB.json',
     'WB.json',
     'WEB.json',
     'YLT.json']




```python
data = {}
for file in onlyfiles:
    with open(f"./data/{file}") as f:
        bible = json.load(f)
    data.update(bible)
```


```python
for key in data.keys():
    print(key)
```

    AKJV
    ASV
    BASICENGLISH
    DARBY
    DOUAYRHEIMS
    EMPHBBL
    KJV
    KJV1900
    LEB
    WB
    WEB
    YLT
    

# General Statistics

Here, we will make a dataframe out of the parsed text to get a better idea of how the information is formatted.  All of the versions will be added and general statistics will be calculated to see how the available Bibles differ from one another.

Some of these statistics include the following:

`char_count` <br>
`word_count` <br>
`punctuation_count` <br>


```python
# Create dataframes out of the Bible versions stored
list_of_df = []
for version in data.keys():
    content = []
    for item in data[version].items():
        ref, text = item

        book = ' '.join(ref.split()[:-1]).lower().replace(' ','_')
        chapter = ref.split()[-1].split(":")[0]
        verse = ref.split()[-1].split(":")[1]
        
        if book == 'psalm':
            book = 'psalms'
        if book == 'song_of_solomon':
            book = 'song_of_songs'

        content.append((version.lower(),book,chapter,verse,text))
    
    df = pd.DataFrame(data=content, columns=['version','book','chapter','verse','text'])
    list_of_df.append(df)

df = pd.concat(list_of_df)
```

Create general word statistics


```python
df['char_count'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))
df['word_density'] = df['char_count'] / (df['word_count']+1)
df['punctuation_count'] = df['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
df['title_word_count'] = df['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
df['upper_case_word_count'] = df['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 372964 entries, 0 to 31101
    Data columns (total 11 columns):
    version                  372964 non-null object
    book                     372964 non-null object
    chapter                  372964 non-null object
    verse                    372964 non-null object
    text                     372964 non-null object
    char_count               372964 non-null int64
    word_count               372964 non-null int64
    word_density             372964 non-null float64
    punctuation_count        372964 non-null int64
    title_word_count         372964 non-null int64
    upper_case_word_count    372964 non-null int64
    dtypes: float64(1), int64(5), object(5)
    memory usage: 34.1+ MB
    


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>version</th>
      <th>book</th>
      <th>chapter</th>
      <th>verse</th>
      <th>text</th>
      <th>char_count</th>
      <th>word_count</th>
      <th>word_density</th>
      <th>punctuation_count</th>
      <th>title_word_count</th>
      <th>upper_case_word_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>akjv</td>
      <td>genesis</td>
      <td>1</td>
      <td>1</td>
      <td>In the beginning God created the heaven and th...</td>
      <td>54</td>
      <td>10</td>
      <td>4.909091</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>akjv</td>
      <td>genesis</td>
      <td>1</td>
      <td>2</td>
      <td>And the earth was without form, and void; and ...</td>
      <td>138</td>
      <td>29</td>
      <td>4.600000</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>akjv</td>
      <td>genesis</td>
      <td>1</td>
      <td>3</td>
      <td>And God said, Let there be light: and there wa...</td>
      <td>54</td>
      <td>11</td>
      <td>4.500000</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>akjv</td>
      <td>genesis</td>
      <td>1</td>
      <td>4</td>
      <td>And God saw the light, that it was good: and G...</td>
      <td>85</td>
      <td>17</td>
      <td>4.722222</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>akjv</td>
      <td>genesis</td>
      <td>1</td>
      <td>5</td>
      <td>And God called the light Day, and the darkness...</td>
      <td>115</td>
      <td>22</td>
      <td>5.000000</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.book.unique()
```




    array(['genesis', 'exodus', 'leviticus', 'numbers', 'deuteronomy',
           'joshua', 'judges', 'ruth', '1_samuel', '2_samuel', '1_kings',
           '2_kings', '1_chronicles', '2_chronicles', 'ezra', 'nehemiah',
           'esther', 'job', 'psalms', 'proverbs', 'ecclesiastes',
           'song_of_songs', 'isaiah', 'jeremiah', 'lamentations', 'ezekiel',
           'daniel', 'hosea', 'joel', 'amos', 'obadiah', 'jonah', 'micah',
           'nahum', 'habakkuk', 'zephaniah', 'haggai', 'zechariah', 'malachi',
           'matthew', 'mark', 'luke', 'john', 'acts', 'romans',
           '1_corinthians', '2_corinthians', 'galatians', 'ephesians',
           'philippians', 'colossians', '1_thessalonians', '2_thessalonians',
           '1_timothy', '2_timothy', 'titus', 'philemon', 'hebrews', 'james',
           '1_peter', '2_peter', '1_john', '2_john', '3_john', 'jude',
           'revelation'], dtype=object)




```python
## Tokenize text and add it as a column in the dataframe
# df['tokenize_verses'] = df['text'].apply(lambda x: th.tokenize_verse(x))
```

### Old Testament Comparison & New Testament Comparision against different versions


```python
aggrts = df.groupby(['version']).agg('sum')
```


```python
aggrts.columns
```




    Index(['char_count', 'word_count', 'word_density', 'punctuation_count',
           'title_word_count', 'upper_case_word_count'],
          dtype='object')




```python
#Graph function to make bar graph
def generate_bar_graph(df=df, metric='', function='', title='Graph'):
    fig = go.Figure([go.Bar(x=[item[0]], y=[item[1]], 
                            name=item[0], showlegend=True) 
                     for item in df[metric].items()
                    ])
    fig.update_layout(title_text=title)
    fig.show()
```


```python
# generate_bar_graph(aggrts, metric='char_count', function='sum')
```


```python
## Graph of metrics sums
columns, index = aggrts.columns, 0 
rows, cols = 2, len(columns)//2

fig, axs = plt.subplots(rows, cols, figsize=(20,15))

for i in range(0,rows):
    for j in range(0,cols):
        
        barplot = sns.barplot(x='version', y=columns[index], data=aggrts.reset_index(),
                    palette="muted", ax=axs[i][j])
        barplot.set_xticklabels(barplot.get_xticklabels(), rotation=90)
        barplot.set_title(label=f'{columns[index]} by version')
        barplot.xaxis.set_label([])
        index += 1

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, 
                    right=0.95, hspace=0.25, wspace=0.35)
```


![png](Bible_Comparison_files/Bible_Comparison_47_0.png)


### Mean metrics


```python
aggrts = df.groupby(['version']).agg('mean')

## Graph of metrics sums
columns, index = aggrts.columns, 0 
rows, cols = 2, len(columns)//2

fig, axs = plt.subplots(rows, cols, figsize=(20,15))

for i in range(0,rows):
    for j in range(0,cols):
        
        barplot = sns.barplot(x='version', y=columns[index], data=aggrts.reset_index(),
                    palette="muted", ax=axs[i][j])
        barplot.set_xticklabels(barplot.get_xticklabels(), rotation=90)
        barplot.set_title(label=f'{columns[index]} by version')
        barplot.xaxis.set_label([])
        index += 1

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
```


![png](Bible_Comparison_files/Bible_Comparison_49_0.png)


### Median metrics


```python
aggrts = df.groupby(['version']).agg('median')

## Graph of metrics sums
columns, index = aggrts.columns, 0 
rows, cols = 2, len(columns)//2

fig, axs = plt.subplots(rows, cols, figsize=(20,15))

for i in range(0,rows):
    for j in range(0,cols):
        
        barplot = sns.barplot(x='version', y=columns[index], data=aggrts.reset_index(),
                    palette="muted", ax=axs[i][j])
        barplot.set_xticklabels(barplot.get_xticklabels(), rotation=90)
        barplot.set_title(label=f'{columns[index]} by version')
        barplot.xaxis.set_label([])
        index += 1

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
```


![png](Bible_Comparison_files/Bible_Comparison_51_0.png)


# Modeling

### TF-IDF Vectorization


```python
## These are functions taken from Flat Iron Curriculum

def count_vectorize(line, vocab=None):

    if vocab:  
        for word in line:
            if word in vocab:
                vocab[word] += 1
            else: 
                vocab[word] = 1
        return vocab
        
    else:
        unique_words = list(set(line))

        text_dict = {i:0 for i in unique_words}
        
        for word in line:
            if word in text_dict:
                text_dict[word] += 1
            else :
                text_dict[word] = 1    
        
        return text_dict

def term_frequency(BoW_dict):
    total_word_count = sum(BoW_dict.values())
    
    for ind, val in BoW_dict.items():
        BoW_dict[ind] = val/ total_word_count
    
    return BoW_dict


def inverse_document_frequency(list_of_dicts):
    vocab_set = set()
    # Iterate through list of dfs and add index to vocab_set
    for d in list_of_dicts:
        for word in d.keys():
            vocab_set.add(word)
    
    # Once vocab set is complete, create an empty dictionary with a key for each word and value of 0.
    full_vocab_dict = {i:0 for i in vocab_set}
    
    # Loop through each word in full_vocab_dict
    for word, val in full_vocab_dict.items():
        docs = 0
        
        # Loop through list of dicts.  Each time a dictionary contains the word, increment docs by 1
        for d in list_of_dicts:
            if word in d:
                docs += 1
        
        # Now that we know denominator for equation, compute and set IDF value for word
        
        full_vocab_dict[word] = np.log((len(list_of_dicts)/ float(docs)))
    
    return full_vocab_dict


def tf_idf(list_of_dicts):
    
    # Create empty dictionary containing full vocabulary of entire corpus
    doc_tf_idf = {}
    idf = inverse_document_frequency(list_of_dicts)
    full_vocab_list = {i:0 for i in list(idf.keys())}
    
    # Create tf-idf list of dictionaries, containing a dictionary that will be updated for each document
    tf_idf_list_of_dicts = []
    
    # Now, compute tf and then use this to compute and set tf-idf values for each document
    for doc in list_of_dicts:
        doc_tf = term_frequency(doc)
        for word in doc_tf:
            doc_tf_idf[word] = doc_tf[word] * idf[word]
        tf_idf_list_of_dicts.append(doc_tf_idf)
    
    return tf_idf_list_of_dicts
```


```python
list_of_verses = df[(df.version=='asv')].text.apply(lambda x: th.tokenize_verse(x))
list_of_verses[:5]
```


```python
vocab_count = {}
for verse in list_of_verses:
     vocab_count = count_vectorize(verse, vocab_count)
        
vocab_count
```


```python
term_freq = term_frequency(vocab_count)
```


```python
# inverse_document_frequency([term_freq])
tf_idf([term_freq])
```


```python
for key in data:
    list_of_verses = list(data[key].values())
    
    vocab_count = {}
    for verse in list_of_verses:
         vocab_count = count_vectorize(verse, vocab_count)
    term_freq = term_frequency(vocab_count)
    tf_idf()
```

### TF-idf vectorizer using sklearn


```python
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
```


```python
# list_of_verses = [' '.join(val) for val in data['KJV'].values()]
list_of_verses = df.loc[(df.version=='kjv')].text # .apply(lambda x: th.tokenize_verse(x))
```


```python
cv = CountVectorizer()
word_count_vector = cv.fit_transform(list_of_verses)
```


```python
word_count_vector.shape
```

__Compute the IDF values__


```python
tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)
```


```python
# print idf values
idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])
 
# sort ascending
idf.sort_values(by=['idf_weights'])
```

__Compute the TFIDF score for the document__


```python
# count matrix
count_vector=cv.transform(list_of_verses)
 
# tf-idf scores
tf_idf_vector=tfidf_transformer.transform(count_vector)
```


```python
feature_names = cv.get_feature_names()
 
#get tfidf vector for first document
first_document_vector=tf_idf_vector[0]
 
#print the scores
df_vec = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"])
df_vec.sort_values(by=["tfidf"],ascending=False)
```

### Word2Vector

Create the dictionary of models to hold the different models


```python
versions_wv_dict = {}
```


```python
for key in data.keys():
    vals = df[(df.version==key.lower())].text.apply(lambda x: th.tokenize_verse(x)).values
    model = Word2Vec(vals, size=100, window=5, min_count=1, workers=4)
    model.train(vals, total_examples=model.corpus_count, epochs=10)
    versions_wv_dict[key] = model.wv
```


```python
words_to_check = ['salvation','resurrection','healing', 'redemption','hope','joy','peace']
```

Find the most similar words


```python
index = 0
for word in words_to_check:
    for key in versions_wv_dict.keys():
        print(f"Version: {key} \t Word: {word}")
        try:
            similar_words = versions_wv_dict[key].most_similar(positive=[word], topn=10)
            for related in similar_words:
                print(related)
            print('\n\n')
        except KeyError:
            print(f"{word} not in the vocabulary")
            print('\n\n')
```

    Version: AKJV 	 Word: salvation
    ('defense', 0.6906821727752686)
    ('strength', 0.6901434063911438)
    ('Father', 0.6790546178817749)
    ('faithfulness', 0.6633484959602356)
    ('mercies', 0.6485440731048584)
    ('prayer', 0.6431960463523865)
    ('hope', 0.62909334897995)
    ('joy', 0.6275311708450317)
    ('goodness', 0.625328779220581)
    ('Lead', 0.6202820539474487)
    
    
    
    Version: ASV 	 Word: salvation
    ('Redeemer', 0.7297897338867188)
    ('lovingkindness', 0.6543046236038208)
    ('Father', 0.6490455865859985)
    ('strength', 0.6484163999557495)
    ('goodness', 0.6420180201530457)
    ('faithfulness', 0.6228456497192383)
    ('fortress', 0.6201319098472595)
    ('dwelling-place', 0.6194751858711243)
    ('righteousness', 0.607792854309082)
    ('rebuke', 0.5848723649978638)
    
    
    
    Version: BASICENGLISH 	 Word: salvation
    ('strength', 0.659477710723877)
    ('reward', 0.6499365568161011)
    ('help', 0.6358147263526917)
    ('grace', 0.5936710834503174)
    ('glory', 0.5879794359207153)
    ('blessing', 0.5870475769042969)
    ('mercy', 0.5826022624969482)
    ('support', 0.5723560452461243)
    ('comfort', 0.5702806711196899)
    ('saviour', 0.5587430000305176)
    
    
    
    Version: DARBY 	 Word: salvation
    ('faithfulness', 0.6988648176193237)
    ('Father', 0.6751481294631958)
    ('Redeemer', 0.6639248132705688)
    ('strength', 0.6634668111801147)
    ('hope', 0.6322900652885437)
    ('goodness', 0.6286040544509888)
    ('glory', 0.6283248066902161)
    ('loving-kindness', 0.6244831085205078)
    ('righteousness', 0.6244112253189087)
    ('mercies', 0.6121140718460083)
    
    
    
    Version: DOUAYRHEIMS 	 Word: salvation
    ('protector', 0.6752961874008179)
    ('justice', 0.6636704206466675)
    ('charity', 0.6633039116859436)
    ('truth', 0.6475796103477478)
    ('glory', 0.64508056640625)
    ('helper', 0.6421048641204834)
    ('Father', 0.6388111114501953)
    ('hope', 0.6374785900115967)
    ('Saviour', 0.6277733445167542)
    ('patience', 0.6162632703781128)
    
    
    
    Version: EMPHBBL 	 Word: salvation
    ('Name', 0.7164942026138306)
    ('Father', 0.6808338165283203)
    ('midst—O', 0.6669315099716187)
    ('lovingkindness', 0.6536811590194702)
    ('compassions', 0.6516642570495605)
    ('Redeemer', 0.6489191055297852)
    ('strength', 0.6450096368789673)
    ('truthfulness', 0.6386401653289795)
    ('foes', 0.6341021060943604)
    ('hope', 0.6334816813468933)
    
    
    
    Version: KJV 	 Word: salvation
    ('defence', 0.6933519840240479)
    ('strength', 0.6864064931869507)
    ('faithfulness', 0.6807496547698975)
    ('goodness', 0.6350321769714355)
    ('mercies', 0.6227455735206604)
    ('joy', 0.6217313408851624)
    ('Father', 0.6187987923622131)
    ('hope', 0.6127660274505615)
    ('righteousness', 0.6116783022880554)
    ('prayer', 0.5998460054397583)
    
    
    
    Version: KJV1900 	 Word: salvation
    ('strength', 0.674288272857666)
    ('Father', 0.6406633257865906)
    ('faithfulness', 0.6321600675582886)
    ('goodness', 0.6317286491394043)
    ('lovingkindness', 0.6254951357841492)
    ('defence', 0.6176563501358032)
    ('righteousness', 0.6172603368759155)
    ('prayer', 0.6145692467689514)
    ('mercies', 0.6086496710777283)
    ('patience', 0.5990892648696899)
    
    
    
    Version: LEB 	 Word: salvation
    ('Father', 0.6733430624008179)
    ('glory', 0.6717056035995483)
    ('holiness', 0.6391419172286987)
    ('willingness', 0.6335363388061523)
    ('joy', 0.624951958656311)
    ('praise', 0.6243896484375)
    ('faithfulness', 0.6215345859527588)
    ('strength', 0.6153789758682251)
    ('soul', 0.6152352094650269)
    ('majesty', 0.6081055402755737)
    
    
    
    Version: WB 	 Word: salvation
    ('defense', 0.689892053604126)
    ('strength', 0.6783117055892944)
    ('Father', 0.6506971120834351)
    ('faithfulness', 0.647454559803009)
    ('fortress', 0.6331539750099182)
    ('righteousness', 0.6256083250045776)
    ('goodness', 0.6236200332641602)
    ('loving-kindness', 0.6221440434455872)
    ('prayers', 0.6164525747299194)
    ('footstool', 0.6159361600875854)
    
    
    
    Version: WEB 	 Word: salvation
    ('faithfulness', 0.6790916323661804)
    ('Father', 0.6678524613380432)
    ('strength', 0.6629630327224731)
    ('goodness', 0.6487554311752319)
    ('hope', 0.6391161680221558)
    ('prayer', 0.636843204498291)
    ('praise', 0.628508985042572)
    ('Savior', 0.6108860969543457)
    ('affliction', 0.6099178791046143)
    ('fortress', 0.5959509015083313)
    
    
    
    Version: YLT 	 Word: salvation
    ('strength', 0.7578702569007874)
    ('excellency', 0.6888039112091064)
    ('zeal', 0.6790338754653931)
    ('kindness', 0.6629945635795593)
    ('Name', 0.6438920497894287)
    ('goodness', 0.6351192593574524)
    ('Redeemer', 0.6335000991821289)
    ('One', 0.6312121748924255)
    ('lowliness', 0.6234654188156128)
    ('habitation', 0.6188845038414001)
    
    
    
    Version: AKJV 	 Word: resurrection
    ('baptism', 0.5698535442352295)
    ('descent', 0.5396053791046143)
    ('uncircumcision', 0.5329654812812805)
    ('Gabbatha', 0.5244396924972534)
    ('appearing', 0.5226025581359863)
    ('Samlah', 0.50875324010849)
    ('grace', 0.5056958794593811)
    ('inspiration', 0.4960835576057434)
    ('hope', 0.49519866704940796)
    ('driving', 0.4949430525302887)
    
    
    
    Version: ASV 	 Word: resurrection
    ('bodies', 0.5921794176101685)
    ('rudiments', 0.5390092134475708)
    ('latchet', 0.5373284220695496)
    ('body', 0.5346174240112305)
    ('error', 0.5140161514282227)
    ('defilements', 0.5106896162033081)
    ('sufferings', 0.5067340135574341)
    ('uncircumcision', 0.5024410486221313)
    ('baptism', 0.4996098279953003)
    ('faith', 0.4966190755367279)
    
    
    
    Version: BASICENGLISH 	 Word: resurrection
    resurrection not in the vocabulary
    
    
    
    Version: DARBY 	 Word: resurrection
    ('baptism', 0.5503225326538086)
    ('scriptures', 0.5464996099472046)
    ('Offspring', 0.5285773277282715)
    ('Christ', 0.5120440721511841)
    ('sufferings', 0.508813202381134)
    ('Howbeit', 0.5039303302764893)
    ('uncircumcision', 0.5028700232505798)
    ('error', 0.5022450089454651)
    ('beginning', 0.5011941194534302)
    ('scourging', 0.5006635785102844)
    
    
    
    Version: DOUAYRHEIMS 	 Word: resurrection
    ('baptism', 0.5731785297393799)
    ('gospel', 0.5627809762954712)
    ('Baptist', 0.5312327742576599)
    ('imprudent', 0.519548773765564)
    ('disputer', 0.5035964250564575)
    ('passion', 0.503126859664917)
    ('Redeemer', 0.5018200874328613)
    ('prophet', 0.4986765384674072)
    ('scripture', 0.49467581510543823)
    ('Spirit', 0.49284666776657104)
    
    
    
    Version: EMPHBBL 	 Word: resurrection
    ('record', 0.5815117359161377)
    ('circumcision', 0.5766996741294861)
    ('charm', 0.5677608251571655)
    ('encouragement', 0.5650389194488525)
    ('testing', 0.5585927963256836)
    ('tribulation', 0.556965172290802)
    ('mockeries', 0.5557900071144104)
    ('hope', 0.5512324571609497)
    ('conclusion', 0.5483005046844482)
    ('sufferings', 0.548071563243866)
    
    
    
    Version: KJV 	 Word: resurrection
    ('effectual', 0.5461595058441162)
    ('fraud', 0.533002495765686)
    ('freewoman', 0.5234930515289307)
    ('En-rogel', 0.5179021954536438)
    ('circumcision', 0.5043548941612244)
    ('uncircumcision', 0.5007048845291138)
    ('faith', 0.4998628795146942)
    ('Meonenim', 0.4993668794631958)
    ('cliff', 0.49587011337280273)
    ('Eglah', 0.49479538202285767)
    
    
    
    Version: KJV1900 	 Word: resurrection
    ('uncircumcision', 0.5716008543968201)
    ('inspiration', 0.556415855884552)
    ('circumcision', 0.5370450615882874)
    ('similitude', 0.5192955732345581)
    ('baptisms', 0.518492579460144)
    ('testator', 0.5105966329574585)
    ('Spirit', 0.504903256893158)
    ('body', 0.5027591586112976)
    ('sadness', 0.4985717236995697)
    ('baptism', 0.4933396875858307)
    
    
    
    Version: LEB 	 Word: resurrection
    ('Ithream', 0.5949523448944092)
    ('declaration', 0.5887891054153442)
    ('quarters', 0.5832883715629578)
    ('prophecy', 0.5823843479156494)
    ('Ethnaim', 0.5806821584701538)
    ('regulation', 0.5787566900253296)
    ('Greek', 0.571338951587677)
    ('baptism', 0.5620875358581543)
    ('rioting', 0.556958794593811)
    ('likeness', 0.5521455407142639)
    
    
    
    Version: WB 	 Word: resurrection
    ('baptism', 0.5264704823493958)
    ('uncircumcision', 0.5237611532211304)
    ('dead', 0.5174623131752014)
    ('Caleb-ephratah', 0.5014854669570923)
    ('Aharah', 0.49066752195358276)
    ('body', 0.4904005229473114)
    ('disease', 0.4822469651699066)
    ('circumcision', 0.4778461456298828)
    ('living', 0.4773455560207367)
    ('appearing', 0.4725053012371063)
    
    
    
    Version: WEB 	 Word: resurrection
    ('oppression', 0.5574181079864502)
    ('uncircumcision', 0.5472358465194702)
    ('dead', 0.5455044507980347)
    ('hearing', 0.4955177903175354)
    ('Adam', 0.49138104915618896)
    ('appointment', 0.48663026094436646)
    ('body', 0.4820703864097595)
    ('holiness', 0.4814220368862152)
    ('Twin', 0.4785504937171936)
    ('Preparation', 0.473297119140625)
    
    
    
    Version: YLT 	 Word: resurrection
    resurrection not in the vocabulary
    
    
    
    Version: AKJV 	 Word: healing
    ('fowl', 0.7460061311721802)
    ('creature', 0.6887866258621216)
    ('whales', 0.6746267676353455)
    ('workmanship', 0.6644667387008667)
    ('winged', 0.6555115580558777)
    ('theirs', 0.649472713470459)
    ('supplies', 0.646450400352478)
    ('sickness', 0.6342107057571411)
    ('binding', 0.6317780017852783)
    ('kind', 0.6163854002952576)
    
    
    
    Version: ASV 	 Word: healing
    ('seedtime', 0.6637653112411499)
    ('calm', 0.6504943370819092)
    ('Kanah', 0.644515335559845)
    ('Beth-hoglah', 0.6265882849693298)
    ('Maralah', 0.6237795352935791)
    ('egresses', 0.6226145029067993)
    ('growth', 0.6188056468963623)
    ('Dabbesheth', 0.6080574989318848)
    ('Clouds', 0.6035979986190796)
    ('Beth-arabah', 0.6032935380935669)
    
    
    
    Version: BASICENGLISH 	 Word: healing
    healing not in the vocabulary
    
    
    
    Version: DARBY 	 Word: healing
    ('felled', 0.7701501250267029)
    ('boot', 0.6851317882537842)
    ('disease', 0.6821376085281372)
    ('artificer', 0.6782914400100708)
    ('soundness', 0.6769883632659912)
    ('bodily', 0.6647394895553589)
    ('exploreth', 0.6580021381378174)
    ('blame', 0.6571840047836304)
    ('manners', 0.6558898091316223)
    ('deserve', 0.6543748378753662)
    
    
    
    Version: DOUAYRHEIMS 	 Word: healing
    ('roasting', 0.6458921432495117)
    ('supporting', 0.6450314521789551)
    ('commending', 0.6393221616744995)
    ('saint', 0.638352632522583)
    ('winged', 0.6269010901451111)
    ('supplieth', 0.6261541247367859)
    ('enduring', 0.6234011650085449)
    ('human', 0.617595911026001)
    ('godliness', 0.615688145160675)
    ('metal', 0.610834538936615)
    
    
    
    Version: EMPHBBL 	 Word: healing
    ('easier', 0.7063479423522949)
    ('fire-flame', 0.7015840411186218)
    ('toiler', 0.691187858581543)
    ('cry—out', 0.6842365860939026)
    ('plumage', 0.6836767196655273)
    ('treasureth', 0.6647881269454956)
    ('eager', 0.6626244187355042)
    ('Raven', 0.6605530977249146)
    ('dangerous', 0.6557062864303589)
    ('marring', 0.652805507183075)
    
    
    
    Version: KJV 	 Word: healing
    ('winged', 0.6529439091682434)
    ('creature', 0.6474382281303406)
    ('whales', 0.6340759992599487)
    ('fowl', 0.624854564666748)
    ('challengeth', 0.5959807634353638)
    ('abominable', 0.5900098085403442)
    ('increasing', 0.587035059928894)
    ('workmanship', 0.585950493812561)
    ('supplieth', 0.5850903391838074)
    ('founder', 0.5702722072601318)
    
    
    
    Version: KJV1900 	 Word: healing
    ('fried', 0.6110835075378418)
    ('challengeth', 0.6024165153503418)
    ('proportion', 0.5991689562797546)
    ('winged', 0.5980425477027893)
    ('workmanship', 0.5934382677078247)
    ('working', 0.5879759788513184)
    ('edifying', 0.5834896564483643)
    ('marrying', 0.5745830535888672)
    ('sickness', 0.572179913520813)
    ('fowl', 0.5706357955932617)
    
    
    
    Version: LEB 	 Word: healing
    ('kin', 0.6424909830093384)
    ('sanctification', 0.6389093399047852)
    ('contradict', 0.6375638842582703)
    ('nonjudgmental', 0.6263329982757568)
    ('impossible', 0.624887228012085)
    ('harpoons', 0.6234760284423828)
    ('profit', 0.6221726536750793)
    ('species', 0.6185725927352905)
    ('godliness', 0.6170463562011719)
    ('cud—it', 0.615998387336731)
    
    
    
    Version: WB 	 Word: healing
    ('supplieth', 0.6607211828231812)
    ('whales', 0.6551347970962524)
    ('workmanship', 0.6441513299942017)
    ('creature', 0.6423292756080627)
    ('sleight', 0.6324074268341064)
    ('fowl', 0.6319109201431274)
    ('passovers', 0.6208651065826416)
    ('acknowledging', 0.6168492436408997)
    ('Zebulunites', 0.6041773557662964)
    ('godliness', 0.6036635041236877)
    
    
    
    Version: WEB 	 Word: healing
    ('overflows', 0.6420405507087708)
    ('produces', 0.6231465935707092)
    ('herb', 0.6215626001358032)
    ('slave-traders', 0.6057488918304443)
    ('herbs', 0.6012747287750244)
    ('charmer', 0.5954864025115967)
    ('jacinth', 0.5920312404632568)
    ('ravenous', 0.5912856459617615)
    ('equipped', 0.5903874635696411)
    ('acre', 0.5856513977050781)
    
    
    
    Version: YLT 	 Word: healing
    ('cheer', 0.5670374035835266)
    ('engrafted', 0.5667877793312073)
    ('Death-shade', 0.560029149055481)
    ('requiring', 0.5508022904396057)
    ('accounted', 0.5339551568031311)
    ('Seeketh', 0.5318214893341064)
    ('sure', 0.529360830783844)
    ('gentle', 0.5284379124641418)
    ('yielding', 0.526393473148346)
    ('forgiving', 0.5209205746650696)
    
    
    
    Version: AKJV 	 Word: redemption
    ('dwellings', 0.6068508625030518)
    ('nostrils', 0.5987982749938965)
    ('posterity', 0.5891108512878418)
    ('earnest', 0.5779989957809448)
    ('temples', 0.5734274983406067)
    ('estimation', 0.5707610845565796)
    ('supply', 0.5704854726791382)
    ('waiting', 0.5695880055427551)
    ('charges', 0.558916449546814)
    ('sorceries', 0.5521333813667297)
    
    
    
    Version: ASV 	 Word: redemption
    ('sack', 0.6243977546691895)
    ('iniquities', 0.6055741310119629)
    ('lives', 0.59648597240448)
    ('glorying', 0.5827105045318604)
    ('thigh', 0.5800126791000366)
    ('arm', 0.5675384402275085)
    ('habitations', 0.5627010464668274)
    ('sacks', 0.5596810579299927)
    ('estimation', 0.5581649541854858)
    ('first-fruits', 0.5575898885726929)
    
    
    
    Version: BASICENGLISH 	 Word: redemption
    redemption not in the vocabulary
    
    
    
    Version: DARBY 	 Word: redemption
    ('hand', 0.5653220415115356)
    ('translation', 0.5220872163772583)
    ('eyes—I', 0.4948361814022064)
    ('tip', 0.483544260263443)
    ('dayspring', 0.4786451458930969)
    ('grace', 0.4726344347000122)
    ('nostrils', 0.47241079807281494)
    ('breath', 0.4723524749279022)
    ('received', 0.4685600996017456)
    ('right', 0.46643000841140747)
    
    
    
    Version: DOUAYRHEIMS 	 Word: redemption
    ('holiness', 0.6841338872909546)
    ('observance', 0.6628677248954773)
    ('Thamnathsare', 0.6478390097618103)
    ('remission', 0.6475743651390076)
    ('revelation', 0.6327745914459229)
    ('purchasing', 0.6326881647109985)
    ('sanctification', 0.6318404078483582)
    ('exhortation', 0.6264432668685913)
    ('verses', 0.6248123645782471)
    ('consolation', 0.6208460330963135)
    
    
    
    Version: EMPHBBL 	 Word: redemption
    ('offences', 0.6216603517532349)
    ('sack', 0.6170405745506287)
    ('price', 0.597383975982666)
    ('harrowing', 0.5964073538780212)
    ('lives', 0.5855458974838257)
    ('Honourable', 0.5739032626152039)
    ('freewill', 0.5671132206916809)
    ('nostrils', 0.5650381445884705)
    ('blood-relation', 0.5642176270484924)
    ('portion', 0.5617647171020508)
    
    
    
    Version: KJV 	 Word: redemption
    ('nostrils', 0.6392040252685547)
    ('infirmities', 0.6263681650161743)
    ('examples', 0.6087809801101685)
    ('dwellingplaces', 0.5948141813278198)
    ('iniquities', 0.57884681224823)
    ('supply', 0.5763782262802124)
    ('lives', 0.5750505924224854)
    ('affliction', 0.5737155675888062)
    ('dough', 0.5718978643417358)
    ('incurable', 0.5702611207962036)
    
    
    
    Version: KJV1900 	 Word: redemption
    ('sack', 0.6376047730445862)
    ('lives', 0.5906141996383667)
    ('infirmities', 0.5799275636672974)
    ('cupbearers', 0.5733983516693115)
    ('neighings', 0.566970944404602)
    ('nostrils', 0.5636470913887024)
    ('belly', 0.5635713338851929)
    ('temples', 0.560318112373352)
    ('sorceries', 0.5568763613700867)
    ('greatness', 0.548956036567688)
    
    
    
    Version: LEB 	 Word: redemption
    ('property', 0.6537986397743225)
    ('tithes', 0.6497071981430054)
    ('delicacies', 0.6457996964454651)
    ('livestock', 0.5919057130813599)
    ('selling', 0.5898218154907227)
    ('aunt', 0.5888298749923706)
    ('lambs', 0.5829599499702454)
    ('separating', 0.5806370973587036)
    ('brethren', 0.5793908834457397)
    ('portion', 0.575655460357666)
    
    
    
    Version: WB 	 Word: redemption
    ('nativity', 0.6709878444671631)
    ('Izrahite', 0.6084942817687988)
    ('foreskin', 0.6017242670059204)
    ('nostrils', 0.6014325618743896)
    ('labors', 0.598110020160675)
    ('Urbane', 0.5963547229766846)
    ('dwellings', 0.5960521697998047)
    ('infirmities', 0.5909709930419922)
    ('Shamhuth', 0.5897839069366455)
    ('portion', 0.580165445804596)
    
    
    
    Version: WEB 	 Word: redemption
    ('money', 0.6278102397918701)
    ('nostrils', 0.6075949668884277)
    ('sack', 0.6031526327133179)
    ('possession', 0.5683251619338989)
    ('treasure', 0.5652270317077637)
    ('Maker', 0.5626639127731323)
    ('thigh', 0.5594103336334229)
    ('patrimony', 0.5482510328292847)
    ('Masters', 0.5391964316368103)
    ('provision', 0.5362163782119751)
    
    
    
    Version: YLT 	 Word: redemption
    ('wickedness', 0.6165899038314819)
    ('bags', 0.573275625705719)
    ('life', 0.5652099847793579)
    ('pleasure', 0.5619951486587524)
    ('souls', 0.560640811920166)
    ('doings', 0.5561497211456299)
    ('fields', 0.5548738241195679)
    ('front-part', 0.5532514452934265)
    ('reward', 0.5521624088287354)
    ('dwellings', 0.5490081906318665)
    
    
    
    Version: AKJV 	 Word: hope
    ('faith', 0.6913934946060181)
    ('grace', 0.6859588623046875)
    ('salvation', 0.62909334897995)
    ('Father', 0.5760574340820312)
    ('confidence', 0.574394702911377)
    ('righteousness', 0.5731847286224365)
    ('goodness', 0.5707083344459534)
    ('truth', 0.5705216526985168)
    ('wickedness', 0.5629252195358276)
    ('delight', 0.5541249513626099)
    
    
    
    Version: ASV 	 Word: hope
    ('reward', 0.6813382506370544)
    ('faith', 0.6465994715690613)
    ('love', 0.6304056644439697)
    ('confidence', 0.6173899173736572)
    ('Father', 0.6005377173423767)
    ('wickedness', 0.5804318189620972)
    ('salvation', 0.5766573548316956)
    ('delight', 0.5755159854888916)
    ('desire', 0.573111891746521)
    ('truth', 0.572870671749115)
    
    
    
    Version: BASICENGLISH 	 Word: hope
    ('faith', 0.6762670278549194)
    ('memory', 0.5829721093177795)
    ('love', 0.5572765469551086)
    ('Rock', 0.5519793629646301)
    ('strength', 0.5498674511909485)
    ('helper', 0.5362248420715332)
    ('help', 0.5259277820587158)
    ('soul', 0.5215785503387451)
    ('delight', 0.5149708986282349)
    ('danger', 0.5141140222549438)
    
    
    
    Version: DARBY 	 Word: hope
    ('salvation', 0.6322901844978333)
    ('grace', 0.5965433716773987)
    ('delight', 0.5926840305328369)
    ('faith', 0.5867316722869873)
    ('love', 0.5781518220901489)
    ('reward', 0.5778546333312988)
    ('encouragement', 0.5755288600921631)
    ('Father', 0.5733417272567749)
    ('boast', 0.5562211871147156)
    ('righteousness', 0.54930579662323)
    
    
    
    Version: DOUAYRHEIMS 	 Word: hope
    ('charity', 0.638981282711029)
    ('salvation', 0.6374785900115967)
    ('patience', 0.6198225021362305)
    ('glory', 0.5816630721092224)
    ('helper', 0.5740219950675964)
    ('rejoice', 0.5691571235656738)
    ('Christ', 0.5656287670135498)
    ('trust', 0.5586538314819336)
    ('labour', 0.5571707487106323)
    ('faith', 0.5531917214393616)
    
    
    
    Version: EMPHBBL 	 Word: hope
    ('faith', 0.7199208736419678)
    ('works', 0.6687690019607544)
    ('sins', 0.6410391330718994)
    ('reward', 0.6350324153900146)
    ('salvation', 0.6334816813468933)
    ('encouragement', 0.6320657730102539)
    ('speech', 0.6307058930397034)
    ('guilt', 0.6275633573532104)
    ('behalf', 0.6237329244613647)
    ('pleasure', 0.6185424327850342)
    
    
    
    Version: KJV 	 Word: hope
    ('confidence', 0.6684165596961975)
    ('faith', 0.6628814935684204)
    ('grace', 0.6433417201042175)
    ('delight', 0.6244306564331055)
    ('salvation', 0.6127660274505615)
    ('love', 0.6075171232223511)
    ('life', 0.6058385372161865)
    ('labour', 0.5831975936889648)
    ('Father', 0.5755956172943115)
    ('wickedness', 0.572636067867279)
    
    
    
    Version: KJV1900 	 Word: hope
    ('faith', 0.6883493661880493)
    ('confidence', 0.6611027717590332)
    ('grace', 0.6513216495513916)
    ('wickedness', 0.6366586685180664)
    ('salvation', 0.5942400693893433)
    ('favour', 0.5797421932220459)
    ('Father', 0.5792982578277588)
    ('distress', 0.5790877342224121)
    ('love', 0.5692068338394165)
    ('righteousness', 0.5650146007537842)
    
    
    
    Version: LEB 	 Word: hope
    ('reward', 0.674074113368988)
    ('faith', 0.6346117258071899)
    ('confidence', 0.6253262162208557)
    ('soul', 0.6251541972160339)
    ('strength', 0.6140597462654114)
    ('salvation', 0.6057880520820618)
    ('Father', 0.6025201082229614)
    ('affliction', 0.5986341238021851)
    ('grace', 0.5900664925575256)
    ('Christ', 0.5756695866584778)
    
    
    
    Version: WB 	 Word: hope
    ('grace', 0.6585297584533691)
    ('faith', 0.6560260057449341)
    ('salvation', 0.5977311134338379)
    ('righteousness', 0.5824210047721863)
    ('delight', 0.5801770687103271)
    ('favor', 0.5790115594863892)
    ('folly', 0.5705003142356873)
    ('love', 0.5646832585334778)
    ('Father', 0.5578517913818359)
    ('life', 0.5553181767463684)
    
    
    
    Version: WEB 	 Word: hope
    ('faith', 0.6489220857620239)
    ('reward', 0.6405150890350342)
    ('salvation', 0.6391161680221558)
    ('Father', 0.6270028352737427)
    ('goodness', 0.6078540086746216)
    ('truth', 0.6067531108856201)
    ('favor', 0.6053868532180786)
    ('strength', 0.6010777950286865)
    ('sight', 0.5972582101821899)
    ('love', 0.5874553918838501)
    
    
    
    Version: YLT 	 Word: hope
    ('confidence', 0.6641277074813843)
    ('faith', 0.6605677604675293)
    ('vows', 0.5973020792007446)
    ('salvation', 0.5953765511512756)
    ('labour', 0.5869835615158081)
    ('reward', 0.5843506455421448)
    ('Father', 0.5821622610092163)
    ('love', 0.5781353712081909)
    ('tribulation', 0.575269341468811)
    ('mind', 0.5723586082458496)
    
    
    
    Version: AKJV 	 Word: joy
    ('gladness', 0.6642091274261475)
    ('goodness', 0.6505391597747803)
    ('beauty', 0.6475476026535034)
    ('rejoice', 0.6465317606925964)
    ('praise', 0.6278682947158813)
    ('salvation', 0.6275311708450317)
    ('sing', 0.6263411045074463)
    ('songs', 0.6075527667999268)
    ('glory', 0.5876967906951904)
    ('rejoicing', 0.5757647752761841)
    
    
    
    Version: ASV 	 Word: joy
    ('gladness', 0.697453498840332)
    ('songs', 0.6323344111442566)
    ('sing', 0.5997511148452759)
    ('prayers', 0.5861483812332153)
    ('rejoice', 0.5813236236572266)
    ('shout', 0.5670571327209473)
    ('salvation', 0.5612910985946655)
    ('patience', 0.5544219017028809)
    ('glory', 0.5517123937606812)
    ('glad', 0.5426195859909058)
    
    
    
    Version: BASICENGLISH 	 Word: joy
    ('sorrow', 0.7028402090072632)
    ('glad', 0.6572127938270569)
    ('grief', 0.6282088756561279)
    ('delight', 0.6110575795173645)
    ('cries', 0.5840381383895874)
    ('grace', 0.5589741468429565)
    ('glory', 0.5496563911437988)
    ('songs', 0.5376495718955994)
    ('strength', 0.5193971395492554)
    ('praise', 0.5152931213378906)
    
    
    
    Version: DARBY 	 Word: joy
    ('gladness', 0.7341700792312622)
    ('shout', 0.651168942451477)
    ('sorrow', 0.6266407370567322)
    ('songs', 0.6211166977882385)
    ('rejoice', 0.6115744113922119)
    ('praise', 0.6072895526885986)
    ('sing', 0.5902212262153625)
    ('aloud', 0.5693449974060059)
    ('filled', 0.5653446316719055)
    ('sighing', 0.5630244016647339)
    
    
    
    Version: DOUAYRHEIMS 	 Word: joy
    ('gladness', 0.7133751511573792)
    ('glory', 0.6080852150917053)
    ('sorrow', 0.6025905609130859)
    ('praise', 0.6006767749786377)
    ('thanksgiving', 0.5979425311088562)
    ('confidence', 0.5943491458892822)
    ('strength', 0.5832391977310181)
    ('salvation', 0.5809145569801331)
    ('majesty', 0.5800549983978271)
    ('terror', 0.5702798962593079)
    
    
    
    Version: EMPHBBL 	 Word: joy
    ('rejoicing', 0.6166073083877563)
    ('gladness', 0.61310875415802)
    ('song', 0.6002568006515503)
    ('shout', 0.5925694704055786)
    ('triumph', 0.5838620662689209)
    ('praise', 0.5803947448730469)
    ('salvation', 0.5679677724838257)
    ('songs', 0.5605205297470093)
    ('exult', 0.5542458295822144)
    ('lovingkindness', 0.5338269472122192)
    
    
    
    Version: KJV 	 Word: joy
    ('gladness', 0.6632997989654541)
    ('songs', 0.6320227384567261)
    ('praise', 0.6254355907440186)
    ('salvation', 0.6217312812805176)
    ('strength', 0.6213817596435547)
    ('rejoice', 0.6059833765029907)
    ('shout', 0.6002283096313477)
    ('rejoicing', 0.5958391427993774)
    ('sing', 0.5913680195808411)
    ('glory', 0.5658438801765442)
    
    
    
    Version: KJV1900 	 Word: joy
    ('gladness', 0.6676827669143677)
    ('praise', 0.5910786986351013)
    ('songs', 0.5773278474807739)
    ('Rend', 0.5759801864624023)
    ('prayers', 0.5714534521102905)
    ('goodness', 0.5706803202629089)
    ('rejoice', 0.5684490203857422)
    ('rejoicing', 0.5683363080024719)
    ('patience', 0.5679694414138794)
    ('sing', 0.5539390444755554)
    
    
    
    Version: LEB 	 Word: joy
    ('sing', 0.6763271689414978)
    ('goodness', 0.666763424873352)
    ('praise', 0.6549416184425354)
    ('shout', 0.6335746049880981)
    ('salvation', 0.624951958656311)
    ('thanksgiving', 0.5909770727157593)
    ('faithfulness', 0.5864930152893066)
    ('rejoice', 0.5841416120529175)
    ('prayers', 0.5817375183105469)
    ('rejoicing', 0.5796658992767334)
    
    
    
    Version: WB 	 Word: joy
    ('gladness', 0.6708859801292419)
    ('rejoice', 0.6432405710220337)
    ('songs', 0.6416829824447632)
    ('praise', 0.6015015840530396)
    ('sing', 0.5850236415863037)
    ('always', 0.5814449787139893)
    ('salvation', 0.5760139226913452)
    ('rejoicing', 0.5590620040893555)
    ('strength', 0.5588894486427307)
    ('glad', 0.5541723966598511)
    
    
    
    Version: WEB 	 Word: joy
    ('gladness', 0.7452341914176941)
    ('songs', 0.6930860877037048)
    ('goodness', 0.6089568734169006)
    ('rejoice', 0.5995226502418518)
    ('strength', 0.5921449661254883)
    ('unceasing', 0.5861456990242004)
    ('praise', 0.5789902210235596)
    ('sing', 0.5602705478668213)
    ('abundance', 0.5548164248466492)
    ('salvation', 0.5510662198066711)
    
    
    
    Version: YLT 	 Word: joy
    ('gladness', 0.6579740047454834)
    ('rejoice', 0.6559630036354065)
    ('rejoicing', 0.6289914846420288)
    ('sympathised', 0.6237829923629761)
    ('songs', 0.5948177576065063)
    ('harp', 0.5752599239349365)
    ('praise', 0.5634298920631409)
    ('singing', 0.5550066232681274)
    ('zeal', 0.5508001446723938)
    ('thanksgiving', 0.5391058325767517)
    
    
    
    Version: AKJV 	 Word: peace
    ('prayer', 0.5813576579093933)
    ('supplication', 0.5617896318435669)
    ('sacrifice', 0.5375949740409851)
    ('sacrifices', 0.5225221514701843)
    ('righteousness', 0.5154324173927307)
    ('admired', 0.4997580647468567)
    ('offerings', 0.4962466359138489)
    ('faithfulness', 0.48696255683898926)
    ('faith', 0.4855688810348511)
    ('freewill', 0.47521185874938965)
    
    
    
    Version: ASV 	 Word: peace
    ('mind', 0.6102413535118103)
    ('supplication', 0.587044358253479)
    ('prayer', 0.566883385181427)
    ('prosperity', 0.5657448172569275)
    ('protection', 0.5364698171615601)
    ('strength', 0.5343252420425415)
    ('answers', 0.5324010848999023)
    ('heart', 0.5322080850601196)
    ('patience', 0.5243277549743652)
    ('ears', 0.520628035068512)
    
    
    
    Version: BASICENGLISH 	 Word: peace
    ('joy', 0.5061101317405701)
    ('quiet', 0.478712260723114)
    ('salvation', 0.47869446873664856)
    ('glad', 0.47602829337120056)
    ('memory', 0.46535903215408325)
    ('righteousness', 0.4648561477661133)
    ('pleasure', 0.46054601669311523)
    ('sorrow', 0.4502052068710327)
    ('faith', 0.4498608708381653)
    ('Nicopolis', 0.44964128732681274)
    
    
    
    Version: DARBY 	 Word: peace
    ('truth', 0.6064374446868896)
    ('righteousness', 0.5743116736412048)
    ('strength', 0.5655370950698853)
    ('mind', 0.5626751184463501)
    ('heart', 0.5591975450515747)
    ('joy', 0.5417192578315735)
    ('rejoice', 0.4976489841938019)
    ('judgment', 0.49579697847366333)
    ('sorrow', 0.4873819649219513)
    ('soul', 0.4865235984325409)
    
    
    
    Version: DOUAYRHEIMS 	 Word: peace
    ('victims', 0.5272877812385559)
    ('prayer', 0.5242756605148315)
    ('praise', 0.5020591020584106)
    ('sacrifices', 0.49537193775177)
    ('offerings', 0.47665464878082275)
    ('truth', 0.47291597723960876)
    ('ears', 0.47219452261924744)
    ('remembrance', 0.47082236409187317)
    ('mercy', 0.46749112010002136)
    ('vows', 0.4670180380344391)
    
    
    
    Version: EMPHBBL 	 Word: peace
    ('faith', 0.5425922274589539)
    ('righteousness', 0.5260785818099976)
    ('pleasure', 0.5136144161224365)
    ('prayer', 0.510442316532135)
    ('heart', 0.49852821230888367)
    ('supplication', 0.49306464195251465)
    ('hearts', 0.47324851155281067)
    ('Name', 0.4725080132484436)
    ('soul', 0.47046607732772827)
    ('nothing', 0.467048704624176)
    
    
    
    Version: KJV 	 Word: peace
    ('sacrifice', 0.5370000600814819)
    ('sacrifices', 0.5362794399261475)
    ('supplication', 0.5173496007919312)
    ('manneh', 0.5128019452095032)
    ('prayer', 0.5098468661308289)
    ('strength', 0.4971729516983032)
    ('prayers', 0.4921721816062927)
    ('meat', 0.49051034450531006)
    ('lives', 0.4901650547981262)
    ('Accept', 0.4896712899208069)
    
    
    
    Version: KJV1900 	 Word: peace
    ('lives', 0.5392175912857056)
    ('prayer', 0.5274885892868042)
    ('sacrifice', 0.5196322202682495)
    ('maneh', 0.5189423561096191)
    ('sacrifices', 0.5142823457717896)
    ('pleasure', 0.5072512030601501)
    ('hearts', 0.5030620098114014)
    ('jeopardy', 0.5012397170066833)
    ('strength', 0.4996640384197235)
    ('righteousness', 0.48721885681152344)
    
    
    
    Version: LEB 	 Word: peace
    ('kindness', 0.571284294128418)
    ('righteousness', 0.5611419677734375)
    ('faithfulness', 0.5407834649085999)
    ('Yah', 0.525558352470398)
    ('Father', 0.5077206492424011)
    ('humble', 0.5001122951507568)
    ('always', 0.4978023171424866)
    ('confidence', 0.49674975872039795)
    ('God—if', 0.49388766288757324)
    ('love', 0.49021732807159424)
    
    
    
    Version: WB 	 Word: peace
    ('pleasure', 0.5426615476608276)
    ('prayer', 0.5387610197067261)
    ('hearts', 0.5244383811950684)
    ('mind', 0.5189257264137268)
    ('ears', 0.5085094571113586)
    ('strength', 0.5046896934509277)
    ('lives', 0.503775417804718)
    ('ways', 0.49680599570274353)
    ('heart', 0.4963448941707611)
    ('righteousness', 0.496275931596756)
    
    
    
    Version: WEB 	 Word: peace
    ('sacrifices', 0.5579452514648438)
    ('prayer', 0.5231970548629761)
    ('thanksgiving', 0.521541953086853)
    ('salvation', 0.49807533621788025)
    ('joy', 0.49239233136177063)
    ('sacrifice', 0.4863533675670624)
    ('offerings', 0.48019835352897644)
    ('Yahweh', 0.4795393943786621)
    ('strength', 0.4745945930480957)
    ('pleasure', 0.4743669629096985)
    
    
    
    Version: YLT 	 Word: peace
    ('kindness', 0.5514033436775208)
    ('righteousness', 0.5293110013008118)
    ('truth', 0.517612636089325)
    ('judgment', 0.4913005232810974)
    ('joy', 0.480244517326355)
    ('none', 0.4784511625766754)
    ('confidence', 0.45147866010665894)
    ('Father', 0.44816091656684875)
    ('blessing', 0.44811975955963135)
    ('heart', 0.4371669292449951)
    
    
    
    

Find the least similar words


```python
# Find the least similar for 
index = 0
for word in words_to_check:
    for key in versions_wv_dict.keys():
        print(f"Version: {key} \t Word: {word}")
        try:
            similar_words = versions_wv_dict[key].most_similar(negative=[word], topn=10)
            for related in similar_words:
                print(related)
            print('\n\n')
        except KeyError:
            print(f"{word} not in the vocabulary")
            print('\n\n')
```

    Version: AKJV 	 Word: salvation
    ('rode', 0.4810250401496887)
    ('hastily', 0.4318574070930481)
    ('Lot', 0.4085233807563782)
    ('eightieth', 0.4013800323009491)
    ('fathoms', 0.39616966247558594)
    ('three', 0.38491812348365784)
    ('six', 0.3827510476112366)
    ('beget', 0.38262319564819336)
    ('eighteen', 0.3796796500682831)
    ('dragging', 0.3783673644065857)
    
    
    
    Version: ASV 	 Word: salvation
    ('colts', 0.5623962879180908)
    ('fathoms', 0.4830929636955261)
    ('rode', 0.41826820373535156)
    ('thirty', 0.4159276783466339)
    ('eighteen', 0.4093627333641052)
    ('six', 0.4076398015022278)
    ('degenerate', 0.4020395576953888)
    ('some', 0.3908286690711975)
    ('three', 0.3863649368286133)
    ('eight', 0.385512113571167)
    
    
    
    Version: BASICENGLISH 	 Word: salvation
    ('Noadiah', 0.5311957001686096)
    ('calmer', 0.4757176339626312)
    ('Bohan', 0.47211846709251404)
    ('Pi-hahiroth', 0.4607517719268799)
    ('Adummim', 0.46028077602386475)
    ('hardest', 0.4586360454559326)
    ('untrained', 0.4500288963317871)
    ('rooting', 0.44728654623031616)
    ('serious-minded', 0.443927139043808)
    ('HOLY', 0.4436378479003906)
    
    
    
    Version: DARBY 	 Word: salvation
    ('rode', 0.4417562484741211)
    ('weaken', 0.4281843900680542)
    ('resembled', 0.42188093066215515)
    ('Zephath', 0.4199889302253723)
    ('pairs', 0.4073335528373718)
    ('pursued', 0.40382570028305054)
    ('arranged', 0.40172284841537476)
    ('provisions', 0.3994988203048706)
    ('three', 0.3939990699291229)
    ('begged', 0.3926966190338135)
    
    
    
    Version: DOUAYRHEIMS 	 Word: salvation
    ('kneaded', 0.44797003269195557)
    ('facing', 0.4052964150905609)
    ('soothsayings', 0.40216949582099915)
    ('ran', 0.3993450999259949)
    ('Jegedelias', 0.3899824023246765)
    ('fetched', 0.38947293162345886)
    ('taking', 0.3835756480693817)
    ('forthwith', 0.3801652193069458)
    ('blowed', 0.3708038330078125)
    ('Gave', 0.367090106010437)
    
    
    
    Version: EMPHBBL 	 Word: salvation
    ('Following', 0.3989902138710022)
    ('gay', 0.396182119846344)
    ('Shaharaim', 0.3897373080253601)
    ('Bedad', 0.3817446529865265)
    ('colts', 0.3804405927658081)
    ('Beth-rehob', 0.37610042095184326)
    ('together—both', 0.37319326400756836)
    ('multitudes', 0.3720899224281311)
    ('eleven', 0.368625283241272)
    ('seventy', 0.3665376305580139)
    
    
    
    Version: KJV 	 Word: salvation
    ('Jechonias', 0.44776642322540283)
    ('fathoms', 0.3867150843143463)
    ('Saddle', 0.384488046169281)
    ('drove', 0.38388413190841675)
    ('sowed', 0.3805497884750366)
    ('rode', 0.37862929701805115)
    ('Igdaliah', 0.3774479031562805)
    ('Gath', 0.3731905221939087)
    ('pursued', 0.37170130014419556)
    ('slew', 0.36937135457992554)
    
    
    
    Version: KJV1900 	 Word: salvation
    ('fathoms', 0.46795254945755005)
    ('fillet', 0.46162357926368713)
    ('eighteen', 0.43620821833610535)
    ('six', 0.42690223455429077)
    ('eight', 0.4227532744407654)
    ('eightieth', 0.41218340396881104)
    ('Igdaliah', 0.39741647243499756)
    ('rode', 0.39683812856674194)
    ('seventy', 0.3937305510044098)
    ('three', 0.38751673698425293)
    
    
    
    Version: LEB 	 Word: salvation
    ('scraping', 0.3998764157295227)
    ('large', 0.39803946018218994)
    ('pollution', 0.384762704372406)
    ('prostitutes', 0.3843066692352295)
    ('expiate', 0.37839823961257935)
    ('undergird', 0.37664368748664856)
    ('lease', 0.37174683809280396)
    ('eating', 0.3666501045227051)
    ('farmers', 0.3606868088245392)
    ('Seirah', 0.36060112714767456)
    
    
    
    Version: WB 	 Word: salvation
    ('relapse', 0.4718858599662781)
    ('three', 0.4103606939315796)
    ('fathoms', 0.4040651023387909)
    ('Lot', 0.4026528298854828)
    ('rode', 0.4001147747039795)
    ('press-vat', 0.3921111822128296)
    ('slew', 0.3862542510032654)
    ('secluded', 0.3829394578933716)
    ('Penuel', 0.3804638683795929)
    ('seventy', 0.37863296270370483)
    
    
    
    Version: WEB 	 Word: salvation
    ('thirty', 0.4277316629886627)
    ('Sceva', 0.4054712951183319)
    ('Pontius', 0.39379221200942993)
    ('Amashsai', 0.38959771394729614)
    ('running', 0.3804185390472412)
    ('Igdaliah', 0.37464025616645813)
    ('pick', 0.3703938126564026)
    ('Succoth', 0.36040788888931274)
    ('happens', 0.3590744137763977)
    ('gleaning', 0.35805821418762207)
    
    
    
    Version: YLT 	 Word: salvation
    ('drew', 0.4647578299045563)
    ('barefoot', 0.4221985340118408)
    ('Ben-Geber', 0.3910841643810272)
    ('Gergesenes', 0.39073801040649414)
    ('risers', 0.38122254610061646)
    ('Shaharaim', 0.3780311048030853)
    ('Shebarim', 0.37515491247177124)
    ('purchase-book', 0.3726308345794678)
    ('scraping', 0.37087973952293396)
    ('Damascenes', 0.36723461747169495)
    
    
    
    Version: AKJV 	 Word: resurrection
    ('rope', 0.45530807971954346)
    ('sheets', 0.39668986201286316)
    ('millions', 0.3894481956958771)
    ('appoint', 0.3736541271209717)
    ('environ', 0.37344053387641907)
    ('Cleanse', 0.34691399335861206)
    ('garlands', 0.34334537386894226)
    ('gird', 0.33148500323295593)
    ('unwalled', 0.3274562954902649)
    ('enclose', 0.3270091116428375)
    
    
    
    Version: ASV 	 Word: resurrection
    ('recompensest', 0.3382337987422943)
    ('prices', 0.33482739329338074)
    ('encounter', 0.33139848709106445)
    ('En-tappuah', 0.31744474172592163)
    ('defendest', 0.3140919506549835)
    ('providest', 0.3116479218006134)
    ('furious', 0.3046836256980896)
    ('grave-clothes', 0.2914612889289856)
    ('coppersmith', 0.2893790006637573)
    ('unwalled', 0.2893696129322052)
    
    
    
    Version: BASICENGLISH 	 Word: resurrection
    resurrection not in the vocabulary
    
    
    
    Version: DARBY 	 Word: resurrection
    ('pottery', 0.4283965229988098)
    ('Choice', 0.40718647837638855)
    ('Daughters', 0.36768269538879395)
    ('accompanying', 0.3570854961872101)
    ('timber', 0.338571697473526)
    ('remorse', 0.33066102862358093)
    ('wine-vat', 0.3302878737449646)
    ('Nehushtan', 0.3200939893722534)
    ('conducting', 0.3156697452068329)
    ('purifier', 0.31502488255500793)
    
    
    
    Version: DOUAYRHEIMS 	 Word: resurrection
    ('compass', 0.3592367470264435)
    ('broke', 0.3583064079284668)
    ('Heap', 0.3547438979148865)
    ('gather', 0.3528134822845459)
    ('square', 0.3527548909187317)
    ('buy', 0.34848034381866455)
    ('get', 0.3470218777656555)
    ('untempered', 0.3459416925907135)
    ('reduce', 0.34562593698501587)
    ('cut', 0.3446334898471832)
    
    
    
    Version: EMPHBBL 	 Word: resurrection
    ('Father､', 0.45021703839302063)
    ('Sanctuary', 0.4149022102355957)
    ('gay', 0.3757973909378052)
    ('Dragged', 0.35032618045806885)
    ('Yahweh—I', 0.34797951579093933)
    ('fast—and', 0.33965396881103516)
    ('thereon', 0.3316836357116699)
    ('whited', 0.32569828629493713)
    ('Baale', 0.324594110250473)
    ('thee､the', 0.3235548734664917)
    
    
    
    Version: KJV 	 Word: resurrection
    ('buy', 0.4063369035720825)
    ('environ', 0.4059462547302246)
    ('Harness', 0.40096235275268555)
    ('sheets', 0.37063777446746826)
    ('assaying', 0.3665090501308441)
    ('get', 0.35870540142059326)
    ('sentest', 0.35798734426498413)
    ('valiantest', 0.3366200029850006)
    ('hew', 0.3360998034477234)
    ('endanger', 0.33136969804763794)
    
    
    
    Version: KJV1900 	 Word: resurrection
    ('environ', 0.38141101598739624)
    ('over', 0.3750203549861908)
    ('disguise', 0.3703744113445282)
    ('Accept', 0.36860477924346924)
    ('millions', 0.3541136682033539)
    ('Reprobate', 0.35087594389915466)
    ('hew', 0.345456063747406)
    ('horsemen', 0.3449432849884033)
    ('stool', 0.3383614420890808)
    ('Make', 0.3273811936378479)
    
    
    
    Version: LEB 	 Word: resurrection
    ('gouge', 0.3889227509498596)
    ('smite', 0.3615344166755676)
    ('woodcutters', 0.36110055446624756)
    ('collected—gold', 0.3327803611755371)
    ('Tartarus', 0.3225049674510956)
    ('whores', 0.31665199995040894)
    ('them—bring', 0.31508708000183105)
    ('Will', 0.2972414195537567)
    ('Nehushtan', 0.296383261680603)
    ('anoint', 0.29263168573379517)
    
    
    
    Version: WB 	 Word: resurrection
    ('stool', 0.40815824270248413)
    ('garlands', 0.40100550651550293)
    ('whited', 0.3686923682689667)
    ('lightest', 0.36549466848373413)
    ('chalk-stones', 0.348856657743454)
    ('about', 0.33845698833465576)
    ('anointest', 0.3195185661315918)
    ('Harness', 0.31863945722579956)
    ('millions', 0.31861409544944763)
    ('Soul', 0.31549859046936035)
    
    
    
    Version: WEB 	 Word: resurrection
    ('whitewashed', 0.3704259395599365)
    ('garlands', 0.3555871546268463)
    ('Hassenaah', 0.33980971574783325)
    ('Take', 0.3379269540309906)
    ('bounds', 0.3323984146118164)
    ('Silver', 0.31947755813598633)
    ('Gather', 0.3186200261116028)
    ('comparable', 0.3185035288333893)
    ('gullible', 0.31770944595336914)
    ('Perfume', 0.31275224685668945)
    
    
    
    Version: YLT 	 Word: resurrection
    resurrection not in the vocabulary
    
    
    
    Version: AKJV 	 Word: healing
    ('Respect', 0.4961960017681122)
    ('Sacrifice', 0.3901236653327942)
    ('prefer', 0.38016360998153687)
    ('enclose', 0.34911513328552246)
    ('Under', 0.34821817278862)
    ('bemoaning', 0.31661030650138855)
    ('First', 0.3084881007671356)
    ('psalmist', 0.29710593819618225)
    ('Therewith', 0.2839084267616272)
    ('cried', 0.2771899104118347)
    
    
    
    Version: ASV 	 Word: healing
    ('First', 0.43111729621887207)
    ('Prove', 0.4258033037185669)
    ('bankers', 0.3746527433395386)
    ('Root', 0.36316800117492676)
    ('errand', 0.3581465482711792)
    ('weareth', 0.357408344745636)
    ('Shouldest', 0.33607667684555054)
    ('turret', 0.33528968691825867)
    ('Hachmonite', 0.31959009170532227)
    ('alienation', 0.31593549251556396)
    
    
    
    Version: BASICENGLISH 	 Word: healing
    healing not in the vocabulary
    
    
    
    Version: DARBY 	 Word: healing
    ('Book', 0.45002084970474243)
    ('sprouted', 0.3970467448234558)
    ('Baalis', 0.35790151357650757)
    ('vote', 0.33797207474708557)
    ('Levitical', 0.32966214418411255)
    ('mountain—that', 0.32342177629470825)
    ('smelled', 0.3215600252151489)
    ('Rage', 0.302581250667572)
    ('instituted', 0.30176296830177307)
    ('leagues', 0.2988993227481842)
    
    
    
    Version: DOUAYRHEIMS 	 Word: healing
    ('nay', 0.36441463232040405)
    ('Laadan', 0.35738468170166016)
    ('treasurers', 0.33528029918670654)
    ('avenge', 0.3176177740097046)
    ('barrels', 0.31594687700271606)
    ('digested', 0.3018631935119629)
    ('outwards', 0.297339528799057)
    ('Fat', 0.29562216997146606)
    ('unwary', 0.2829996347427368)
    ('Launch', 0.2823798656463623)
    
    
    
    Version: EMPHBBL 	 Word: healing
    ('observer', 0.46749481558799744)
    ('Yahweh—I', 0.4436132609844208)
    ('people—unto', 0.42617619037628174)
    ('exceed', 0.4217985272407532)
    ('Gad—even', 0.41944628953933716)
    ('Add', 0.41328030824661255)
    ('Amorites—the', 0.4004979133605957)
    ('seed—so', 0.3941889703273773)
    ('stationing', 0.3897295892238617)
    ('us—seven', 0.3879580497741699)
    
    
    
    Version: KJV 	 Word: healing
    ('Accept', 0.3002156615257263)
    ('Respect', 0.2947801351547241)
    ('Enviest', 0.2764522433280945)
    ('Vow', 0.27014651894569397)
    ('garlands', 0.26911023259162903)
    ('slideth', 0.2670074999332428)
    ('Median', 0.26506903767585754)
    ('Cis', 0.2609466314315796)
    ('First', 0.2605668306350708)
    ('Sceptre', 0.25905030965805054)
    
    
    
    Version: KJV1900 	 Word: healing
    ('Rooted', 0.37943679094314575)
    ('afflictest', 0.32333552837371826)
    ('travailest', 0.32244688272476196)
    ('inclose', 0.31816238164901733)
    ('Conscience', 0.30486124753952026)
    ('instructors', 0.2975718080997467)
    ('Fifteen', 0.29654741287231445)
    ('Respect', 0.2964320182800293)
    ('Accept', 0.27899405360221863)
    ('First', 0.2640380263328552)
    
    
    
    Version: LEB 	 Word: healing
    ('Kiss', 0.4240948557853699)
    ('chopped', 0.3869502544403076)
    ('begotten', 0.37292343378067017)
    ('persevered', 0.3719053566455841)
    ('Drink—and', 0.33223336935043335)
    ('goad', 0.3314743638038635)
    ('from—but', 0.331034392118454)
    ('Buzite', 0.3267704248428345)
    ('live—and', 0.3177033066749573)
    ('Bow', 0.310732901096344)
    
    
    
    Version: WB 	 Word: healing
    ('amerce', 0.40144452452659607)
    ('bemoaning', 0.39094310998916626)
    ('Lovest', 0.3612312972545624)
    ('Bring', 0.34501492977142334)
    ('Respect', 0.3422240912914276)
    ('Shear-jashub', 0.33983123302459717)
    ('millions', 0.3359716534614563)
    ('Rooted', 0.33336642384529114)
    ('psalmist', 0.3326829671859741)
    ('lightest', 0.32753151655197144)
    
    
    
    Version: WEB 	 Word: healing
    ('bemoaning', 0.4260721504688263)
    ('Korahite', 0.35242968797683716)
    ('Hassenaah', 0.35096773505210876)
    ('Root', 0.3507317900657654)
    ('greyhound', 0.33230164647102356)
    ('Zuriel', 0.32315415143966675)
    ('clasp', 0.31990206241607666)
    ('Matrites', 0.3058241903781891)
    ('sworn', 0.29160261154174805)
    ('Ammi', 0.28857558965682983)
    
    
    
    Version: YLT 	 Word: healing
    ('severest', 0.3834574818611145)
    ('Hands', 0.3758322298526764)
    ('memorials', 0.37452614307403564)
    ('chief', 0.3674362003803253)
    ('Shimeam', 0.3671637177467346)
    ('glide', 0.3578440248966217)
    ('elders', 0.3432586193084717)
    ('Aaron', 0.3410062789916992)
    ('sympathise', 0.3341488838195801)
    ('Sacrifices', 0.32645758986473083)
    
    
    
    Version: AKJV 	 Word: redemption
    ('Be', 0.3446905016899109)
    ('Have', 0.3373112380504608)
    ('Respect', 0.3364713490009308)
    ('Read', 0.3330039083957672)
    ('Days', 0.3226802945137024)
    ('Shall', 0.31753408908843994)
    ('Kiss', 0.3172856569290161)
    ('Skin', 0.31046760082244873)
    ('herewith', 0.3099387586116791)
    ('entice', 0.30740368366241455)
    
    
    
    Version: ASV 	 Word: redemption
    ('Enough', 0.3715519905090332)
    ('Shouldest', 0.368008553981781)
    ('Upright', 0.34597158432006836)
    ('Saw', 0.343355655670166)
    ('valiantest', 0.32378876209259033)
    ('Think', 0.3218975365161896)
    ('Verily', 0.3130589425563812)
    ('Gashmu', 0.31227049231529236)
    ('Separate', 0.3122190833091736)
    ('Peradventure', 0.29776495695114136)
    
    
    
    Version: BASICENGLISH 	 Word: redemption
    redemption not in the vocabulary
    
    
    
    Version: DARBY 	 Word: redemption
    ('blew', 0.40178364515304565)
    ('home—on', 0.3823736310005188)
    ('refusing', 0.37841665744781494)
    ('thither', 0.36636748909950256)
    ('clamour', 0.3658725917339325)
    ('hither', 0.36241286993026733)
    ('howl', 0.3531901240348816)
    ('Shittim', 0.3472943603992462)
    ('scribes', 0.3462775945663452)
    ('dispersed', 0.34468141198158264)
    
    
    
    Version: DOUAYRHEIMS 	 Word: redemption
    ('nay', 0.475046843290329)
    ('thoughtest', 0.3892492353916168)
    ('unwary', 0.3819354176521301)
    ('blowed', 0.37468862533569336)
    ('Maserephoth', 0.3699297606945038)
    ('entice', 0.3198927640914917)
    ('Bow', 0.31749075651168823)
    ('runnest', 0.3141669034957886)
    ('Launch', 0.3127168118953705)
    ('protecteth', 0.3102331757545471)
    
    
    
    Version: EMPHBBL 	 Word: redemption
    ('bearer', 0.40510499477386475)
    ('myrtletrees', 0.40115058422088623)
    ('kill—and', 0.3894929587841034)
    ('lengtheneth', 0.3869199752807617)
    ('therethou', 0.373704195022583)
    ('thee—dead', 0.36571282148361206)
    ('thankfully', 0.36005717515945435)
    ('shouted', 0.35518985986709595)
    ('Scribes', 0.35223230719566345)
    ('aloud', 0.3502339720726013)
    
    
    
    Version: KJV 	 Word: redemption
    ('valiantest', 0.3843664824962616)
    ('Gashmu', 0.36900222301483154)
    ('Peradventure', 0.3357170820236206)
    ('imprisoned', 0.3307116627693176)
    ('Kiss', 0.32849204540252686)
    ('Saw', 0.32071059942245483)
    ('Report', 0.3192623257637024)
    ('Gather', 0.29156067967414856)
    ('Must', 0.28799110651016235)
    ('behoved', 0.2845098078250885)
    
    
    
    Version: KJV1900 	 Word: redemption
    ('Naked', 0.360515296459198)
    ('Separate', 0.3153049051761627)
    ('Have', 0.3059978783130646)
    ('fleeing', 0.3051995635032654)
    ('ailed', 0.29067933559417725)
    ('Gather', 0.2896132469177246)
    ('behoved', 0.2874443233013153)
    ('Are', 0.28736376762390137)
    ('thither', 0.2815813422203064)
    ('deliveredst', 0.2705468237400055)
    
    
    
    Version: LEB 	 Word: redemption
    ('Face', 0.5699044466018677)
    ('joking', 0.4290159344673157)
    ('Kiss', 0.38765862584114075)
    ('totter', 0.3646993339061737)
    ('Find', 0.36339396238327026)
    ('Presumptuously', 0.33530309796333313)
    ('excused', 0.33103975653648376)
    ('Declaration', 0.3248722553253174)
    ('misrepresents', 0.3174785375595093)
    ('came—and', 0.31585273146629333)
    
    
    
    Version: WB 	 Word: redemption
    ('Rhoda', 0.4298315644264221)
    ('Bid', 0.33055317401885986)
    ('Days', 0.32610195875167847)
    ('treachery', 0.30740514397621155)
    ('Saw', 0.30630335211753845)
    ('Kiss', 0.2998111844062805)
    ('lightest', 0.29401129484176636)
    ('Read', 0.2887953817844391)
    ('thither', 0.2871723473072052)
    ('False', 0.28660133481025696)
    
    
    
    Version: WEB 	 Word: redemption
    ('annoyed', 0.47268620133399963)
    ('Gashmu', 0.37969204783439636)
    ('Pontius', 0.3774368464946747)
    ('bemoaning', 0.3304442763328552)
    ('Being', 0.3106987774372101)
    ('naturally', 0.3087327778339386)
    ('terribly', 0.3014264404773712)
    ('Streams', 0.2974667549133301)
    ('exclude', 0.29623037576675415)
    ('Tiberius', 0.29532793164253235)
    
    
    
    Version: YLT 	 Word: redemption
    ('Meshach', 0.4166427552700043)
    ('Shadrach', 0.3961330056190491)
    ('Seal', 0.39136970043182373)
    ('Abed-Nego', 0.3768300414085388)
    ('loud', 0.37541401386260986)
    ('Another', 0.35365113615989685)
    ('Ah', 0.34332913160324097)
    ('Elijah', 0.3409213721752167)
    ('Accuse', 0.32544565200805664)
    ('Swallowed', 0.3204982578754425)
    
    
    
    Version: AKJV 	 Word: hope
    ('Midian', 0.49207842350006104)
    ('fetched', 0.4744229316711426)
    ('unwalled', 0.4721547067165375)
    ('took', 0.4622998833656311)
    ('Lot', 0.45792922377586365)
    ('guard', 0.45218420028686523)
    ('Shittim', 0.45206353068351746)
    ('Pul', 0.45195406675338745)
    ('Jephthah', 0.45160233974456787)
    ('couple', 0.4415198564529419)
    
    
    
    Version: ASV 	 Word: hope
    ('fetched', 0.5228563547134399)
    ('blew', 0.4903407394886017)
    ('drew', 0.48266541957855225)
    ('Bohan', 0.4556390345096588)
    ('ran', 0.44670766592025757)
    ('took', 0.4437701106071472)
    ('Neco', 0.4266064465045929)
    ('Succoth', 0.40933293104171753)
    ('brake', 0.40823036432266235)
    ('Zobah', 0.40790411829948425)
    
    
    
    Version: BASICENGLISH 	 Word: hope
    ('smelling', 0.5098146200180054)
    ('Noadiah', 0.5025035738945007)
    ('carbuncle', 0.47115588188171387)
    ('pygarg', 0.458976686000824)
    ('shut-in', 0.4586600065231323)
    ('Lydia', 0.4546763300895691)
    ('onto', 0.45396313071250916)
    ('parting', 0.45356911420822144)
    ('Tibni', 0.45277807116508484)
    ('angrily', 0.4484134018421173)
    
    
    
    Version: DARBY 	 Word: hope
    ('Bohan', 0.5249017477035522)
    ('Midian', 0.4783170521259308)
    ('fetched', 0.4763150215148926)
    ('Zair', 0.470653235912323)
    ('pairs', 0.4680037796497345)
    ('Aram-naharaim', 0.464621901512146)
    ('Ithamar', 0.439270555973053)
    ('captains', 0.43685758113861084)
    ('Barak', 0.42946261167526245)
    ('Abimelech', 0.4287165403366089)
    
    
    
    Version: DOUAYRHEIMS 	 Word: hope
    ('rook', 0.4615001976490021)
    ('soldiers', 0.44642019271850586)
    ('greeting', 0.41671299934387207)
    ('centurions', 0.41257140040397644)
    ('Run', 0.4118955135345459)
    ('Adali', 0.3948255777359009)
    ('Baruch', 0.38902658224105835)
    ('Balaam', 0.3852579593658447)
    ('presents', 0.38426047563552856)
    ('ran', 0.383870005607605)
    
    
    
    Version: EMPHBBL 	 Word: hope
    ('prayeth', 0.4384138584136963)
    ('hastily—unto', 0.42780202627182007)
    ('Shadrach', 0.4242163300514221)
    ('Pul', 0.4224875569343567)
    ('Jezreel—the', 0.4018869996070862)
    ('servants—ships', 0.39754122495651245)
    ('feIlowship', 0.392262727022171)
    ('Meshach', 0.3895570635795593)
    ('Abed-nego', 0.38955238461494446)
    ('took', 0.38794779777526855)
    
    
    
    Version: KJV 	 Word: hope
    ('Jechonias', 0.5181702971458435)
    ('valiantest', 0.4755672514438629)
    ('Pul', 0.4664217233657837)
    ('fetched', 0.44454920291900635)
    ('Archi', 0.4375878572463989)
    ('soldiers', 0.43734803795814514)
    ('Amminadib', 0.43670448660850525)
    ('blew', 0.435743510723114)
    ('guard', 0.43177586793899536)
    ('Succoth', 0.4314931631088257)
    
    
    
    Version: KJV1900 	 Word: hope
    ('Gather', 0.4668906331062317)
    ('Pul', 0.46110013127326965)
    ('fetched', 0.4604645073413849)
    ('tares', 0.45795243978500366)
    ('soldiers', 0.44389230012893677)
    ('Jechonias', 0.4363596439361572)
    ('Bohan', 0.4306616187095642)
    ('Shittim', 0.42187899351119995)
    ('unwalled', 0.4211677312850952)
    ('embraced', 0.41879063844680786)
    
    
    
    Version: LEB 	 Word: hope
    ('Bohan', 0.5151016712188721)
    ('about—Jesus', 0.45039933919906616)
    ('scraping', 0.44939297437667847)
    ('military', 0.44565534591674805)
    ('lured', 0.418581485748291)
    ('carrying', 0.4151046872138977)
    ('princely', 0.4092057943344116)
    ('Bring', 0.4042856693267822)
    ('billowing', 0.38962483406066895)
    ('Shear-Jashub', 0.38593757152557373)
    
    
    
    Version: WB 	 Word: hope
    ('Midian', 0.5385582447052002)
    ('view', 0.4864436388015747)
    ('unwalled', 0.46082809567451477)
    ('Pul', 0.45533978939056396)
    ('Abed-nego', 0.45493030548095703)
    ('Shadrach', 0.45155924558639526)
    ('explore', 0.44918888807296753)
    ('Succoth', 0.44250601530075073)
    ('dromedaries', 0.4402700960636139)
    ('Archi', 0.4400841295719147)
    
    
    
    Version: WEB 	 Word: hope
    ('blew', 0.4745093584060669)
    ('Answerable', 0.4697504937648773)
    ('Zobah', 0.4543134570121765)
    ('Harness', 0.45411646366119385)
    ('Abiram', 0.432586669921875)
    ('commanding', 0.4325329065322876)
    ('Shan', 0.43074408173561096)
    ('pick', 0.4291604459285736)
    ('Succoth', 0.4269307851791382)
    ('Dathan', 0.42547717690467834)
    
    
    
    Version: YLT 	 Word: hope
    ('taketh', 0.4669586420059204)
    ('Ithamar', 0.4179839491844177)
    ('Hadadezer', 0.4016076922416687)
    ('gathereth', 0.39990919828414917)
    ('Midian', 0.39935722947120667)
    ('Seal', 0.3992306590080261)
    ('Bohan', 0.3988627791404724)
    ('Phichol', 0.3925052285194397)
    ('Meshach', 0.39044517278671265)
    ('brazen', 0.3892054557800293)
    
    
    
    Version: AKJV 	 Word: joy
    ('Zalaph', 0.4230109751224518)
    ('centurion', 0.3932855427265167)
    ('Again', 0.3657093942165375)
    ('Balaam', 0.3584592938423157)
    ('Pontius', 0.3581732213497162)
    ('Iscariot', 0.34612324833869934)
    ('Devil', 0.3458903431892395)
    ('Afterward', 0.34554657340049744)
    ('messengers', 0.34268295764923096)
    ('Abinoam', 0.3386809229850769)
    
    
    
    Version: ASV 	 Word: joy
    ('farther', 0.38648706674575806)
    ('Hammedatha', 0.3536342978477478)
    ('Bid', 0.3478548526763916)
    ('Ai', 0.34755730628967285)
    ('reproveth', 0.3385721743106842)
    ('aside', 0.3379250168800354)
    ('named', 0.3371739983558655)
    ('along', 0.33563679456710815)
    ('Shaphan', 0.33524829149246216)
    ('thence', 0.3234221935272217)
    
    
    
    Version: BASICENGLISH 	 Word: joy
    ('two-edged', 0.4316991865634918)
    ('roughly', 0.4234955906867981)
    ('Eldad', 0.4196682572364807)
    ('signing', 0.4049731492996216)
    ('Has', 0.3995792865753174)
    ('Goliath', 0.38545483350753784)
    ('talkers', 0.37140321731567383)
    ('Cherith', 0.3683128356933594)
    ('Tigris', 0.36075878143310547)
    ('Tekel', 0.35256683826446533)
    
    
    
    Version: DARBY 	 Word: joy
    ('brothers', 0.3708702623844147)
    ('Siloam', 0.3677557110786438)
    ('pushing', 0.355540931224823)
    ('adjudged', 0.3482961654663086)
    ('Judas', 0.34753304719924927)
    ('Answerest', 0.34169095754623413)
    ('Pontius', 0.33333128690719604)
    ('circumstance', 0.33156919479370117)
    ('self-made', 0.3274837136268616)
    ('Mede', 0.32635945081710815)
    
    
    
    Version: DOUAYRHEIMS 	 Word: joy
    ('getting', 0.37640056014060974)
    ('blowed', 0.37425947189331055)
    ('greeting', 0.3617491126060486)
    ('Gave', 0.35630595684051514)
    ('Setim', 0.3475193381309509)
    ('Anath', 0.3421894311904907)
    ('Hand', 0.3243263363838196)
    ('unrevenged', 0.32405680418014526)
    ('Reign', 0.3240510821342468)
    ('blaze', 0.3231602907180786)
    
    
    
    Version: EMPHBBL 	 Word: joy
    ('woman—her', 0.3784313201904297)
    ('Unfruitful', 0.33381956815719604)
    ('bitten', 0.3290329575538635)
    ('gay', 0.3290104269981384)
    ('behind', 0.32126545906066895)
    ('Following', 0.3175968527793884)
    ('Shaharaim', 0.3022339940071106)
    ('wielded', 0.29631417989730835)
    ('other', 0.2942177653312683)
    ('brethren—Simon', 0.2877272963523865)
    
    
    
    Version: KJV 	 Word: joy
    ('Amoz', 0.40928125381469727)
    ('Zalaph', 0.39399611949920654)
    ('James', 0.39204704761505127)
    ('Agagite', 0.39179953932762146)
    ('Urijah', 0.39142727851867676)
    ('Hammedatha', 0.3877439498901367)
    ('Judas', 0.38543838262557983)
    ('Koz', 0.3841457962989807)
    ('Simon', 0.3817477822303772)
    ('Again', 0.37495437264442444)
    
    
    
    Version: KJV1900 	 Word: joy
    ('Igdaliah', 0.4220621883869171)
    ('Zalaph', 0.40357255935668945)
    ('Bartimaeus', 0.4009001851081848)
    ('Bohan', 0.36217567324638367)
    ('Amoz', 0.3612246513366699)
    ('Maaleh-acrabbim', 0.3602904677391052)
    ('lintel', 0.35478872060775757)
    ('Hammedatha', 0.35320472717285156)
    ('Timaeus', 0.345843106508255)
    ('Bosor', 0.34407132863998413)
    
    
    
    Version: LEB 	 Word: joy
    ('about—Jesus', 0.3961271643638611)
    ('After', 0.39248713850975037)
    ('ordered', 0.37992727756500244)
    ('Hang', 0.3586556017398834)
    ('instructing', 0.356475293636322)
    ('lease', 0.356323778629303)
    ('behind', 0.3474089503288269)
    ('Iscariot', 0.34694045782089233)
    ('Sceva', 0.3408033549785614)
    ('Last', 0.3373778462409973)
    
    
    
    Version: WB 	 Word: joy
    ('Hammedatha', 0.426765501499176)
    ('Micaiah', 0.37093937397003174)
    ('Tekoites', 0.3673221468925476)
    ('centurion', 0.36724382638931274)
    ('Bid', 0.3664802610874176)
    ('Balaam', 0.3520904779434204)
    ('Penuel', 0.34898558259010315)
    ('Abinoam', 0.34793221950531006)
    ('Zalaph', 0.345363974571228)
    ('repaired', 0.3442457318305969)
    
    
    
    Version: WEB 	 Word: joy
    ('hitherto', 0.42566174268722534)
    ('disregards', 0.39899373054504395)
    ('Igdaliah', 0.3735809922218323)
    ('Gera', 0.3655436336994171)
    ('Nun', 0.3576081693172455)
    ('Bohan', 0.3323900103569031)
    ('named', 0.3320700526237488)
    ('Hammedatha', 0.3298252820968628)
    ('Matrites', 0.32962745428085327)
    ('Beor', 0.32718512415885925)
    
    
    
    Version: YLT 	 Word: joy
    ('Agagite', 0.3906688690185547)
    ('Hammedatha', 0.34860414266586304)
    ('wine-fat', 0.3431183695793152)
    ('Igdaliah', 0.3301529586315155)
    ('Simon', 0.3268023133277893)
    ('mightest', 0.3142525851726532)
    ('purahs', 0.312066912651062)
    ('taketh', 0.310587078332901)
    ('Iscariot', 0.3080643117427826)
    ('draweth', 0.30738702416419983)
    
    
    
    Version: AKJV 	 Word: peace
    ('mustered', 0.44132012128829956)
    ('eighteen', 0.396056592464447)
    ('Zoar', 0.3861558437347412)
    ("potters'clay", 0.37925344705581665)
    ('Shemer', 0.37143030762672424)
    ('homers', 0.36602962017059326)
    ('defended', 0.3658909201622009)
    ('exacted', 0.3637966513633728)
    ('smooth', 0.3553330600261688)
    ('Habaziniah', 0.3520963191986084)
    
    
    
    Version: ASV 	 Word: peace
    ('hunter', 0.41980496048927307)
    ('Wild', 0.4054712951183319)
    ('employment', 0.4011909067630768)
    ('Salu', 0.3949466049671173)
    ('MYSTERY', 0.38923096656799316)
    ('NAZARETH', 0.387822687625885)
    ('weakeneth', 0.37409794330596924)
    ('JEWS', 0.37222379446029663)
    ('whereupon', 0.3648197650909424)
    ('KING', 0.3597225546836853)
    
    
    
    Version: BASICENGLISH 	 Word: peace
    ('priced', 0.4708808958530426)
    ('Watching', 0.4513012766838074)
    ('Opening', 0.440825492143631)
    ('folds', 0.4385049343109131)
    ('pygarg', 0.4370206892490387)
    ('signing', 0.4365795850753784)
    ('talkers', 0.431449830532074)
    ('Went', 0.4249633550643921)
    ('Bohan', 0.4227237105369568)
    ('hardest', 0.4210957884788513)
    
    
    
    Version: DARBY 	 Word: peace
    ('brothers', 0.4152265191078186)
    ('sultry', 0.3787975311279297)
    ('Earth', 0.35718274116516113)
    ('gluttons', 0.3558571934700012)
    ('Canaanitish', 0.3550238013267517)
    ('buyest', 0.35053467750549316)
    ('Chebar—their', 0.3482397794723511)
    ('Fifteen', 0.3479272723197937)
    ('Jerubbesheth', 0.3414323925971985)
    ('material', 0.3408118486404419)
    
    
    
    Version: DOUAYRHEIMS 	 Word: peace
    ('forty-Ave', 0.41650065779685974)
    ('greeting', 0.40801578760147095)
    ('stoutly', 0.39814406633377075)
    ('quails', 0.38554444909095764)
    ('drieth', 0.3807411789894104)
    ('Against', 0.37986186146736145)
    ('sealeth', 0.3760605752468109)
    ('Sehon', 0.3624003529548645)
    ('Pontus', 0.35307443141937256)
    ('Baalhasor', 0.3527289628982544)
    
    
    
    Version: EMPHBBL 	 Word: peace
    ('another—a', 0.4471686780452728)
    ('outwards', 0.43899810314178467)
    ('Babylonia', 0.43437570333480835)
    ('woman—her', 0.4172271192073822)
    ('Forgiving', 0.4156958758831024)
    ('courtwas', 0.4095827341079712)
    ('battle—the', 0.3877689242362976)
    ('Ben-abinadab', 0.38561320304870605)
    ('facing', 0.3820313811302185)
    ('Like', 0.3810928463935852)
    
    
    
    Version: KJV 	 Word: peace
    ('overthrew', 0.47412407398223877)
    ('Asenath', 0.40768927335739136)
    ('mustered', 0.4018058180809021)
    ('Horite', 0.3611135482788086)
    ('Behind', 0.35419178009033203)
    ('Elead', 0.3493858575820923)
    ('Raamses', 0.34801387786865234)
    ('Chushan-rishathaim', 0.3466516137123108)
    ('Dodo', 0.34599626064300537)
    ('Arad', 0.3453306555747986)
    
    
    
    Version: KJV1900 	 Word: peace
    ('Hammedatha', 0.4304769039154053)
    ('Canaanitess', 0.4157329797744751)
    ('Asenath', 0.39826324582099915)
    ('Barachel', 0.3937280774116516)
    ('Agagite', 0.3931020498275757)
    ('mustered', 0.38711032271385193)
    ('Zalaph', 0.38197845220565796)
    ('Bedad', 0.3759615421295166)
    ('Igdaliah', 0.3736582398414612)
    ('Tender', 0.3730365037918091)
    
    
    
    Version: LEB 	 Word: peace
    ('effectiveness', 0.3968130946159363)
    ('carries', 0.38770267367362976)
    ('Into', 0.3865503668785095)
    ('instructing', 0.3757466971874237)
    ('foul-smelling', 0.35935714840888977)
    ('took', 0.33441537618637085)
    ('large', 0.33378922939300537)
    ('stamped', 0.33370089530944824)
    ('spouted', 0.3319495618343353)
    ('Like', 0.33192235231399536)
    
    
    
    Version: WB 	 Word: peace
    ('mustered', 0.4606991410255432)
    ('amongst', 0.4381585717201233)
    ('Igdaliah', 0.41525498032569885)
    ('Salu', 0.4152001142501831)
    ('Upon', 0.4093872308731079)
    ('Raging', 0.39753952622413635)
    ('Achor', 0.39526796340942383)
    ('looking-glasses', 0.39169368147850037)
    ('KINGS', 0.3901522159576416)
    ('Hammedatha', 0.3866935968399048)
    
    
    
    Version: WEB 	 Word: peace
    ('habitable', 0.44607025384902954)
    ('drives', 0.42839524149894714)
    ('Igdaliah', 0.405337929725647)
    ('Rezeph', 0.40275609493255615)
    ('hitherto', 0.3998388648033142)
    ('Sceva', 0.39576905965805054)
    ('KING', 0.3849855065345764)
    ('overthrew', 0.3820037245750427)
    ('Like', 0.3726409673690796)
    ('Gader', 0.36962461471557617)
    
    
    
    Version: YLT 	 Word: peace
    ('Seal', 0.3995693325996399)
    ('directing', 0.3753747344017029)
    ('slaughter-weapon', 0.37446129322052)
    ('Laying', 0.36015915870666504)
    ('consecrateth', 0.3558810353279114)
    ('Shut', 0.3512098789215088)
    ('Ithamar', 0.3493500351905823)
    ('guided', 0.34639835357666016)
    ('Bar-Jona', 0.34296715259552)
    ('steep', 0.33716773986816406)
    
    
    
    

## Word Cloud Representation

In this section, we have code to create a word cloud representation of books within the specified Bible.  This code can be ran on entire versions, to get further insight into similarities between different different versions and books.  I highly recommended going book by book.  Loading entire versions results in long load times.


```python
from wordcloud import WordCloud, STOPWORDS

def get_stop_words(file_path=''):
    
    complete_stoplist = list(STOPWORDS) + list(nltk.corpus.stopwords.words('english'))
    
    if file_path:
        with open(file_path, 'r') as f:
            eliz_stopwords = f.readlines()
            
        eliz_stopwords = [word.strip() for word in eliz_stopwords]
        complete_stoplist += eliz_stopwords
    
    return set(complete_stoplist)
```


```python
def generate_word_cloud(df=None, col='', stopwords=None, save_disp_flag=False, img_width=800, img_height=800):
    """
        Function to generate word cloud. This function generates the cloud and stores it in a file
        within the current directory
    """
    comment_words = ' '

    for val in df[col]:
        tokens = [token for token in th.tokenize_verse(val)]

        for words in tokens:
            comment_words = comment_words + words + ' '

    wordcloud = WordCloud(width = img_width, height = img_height, 
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(comment_words)
    
    return wordcloud.to_image()
```

code to create zoomable static image


```python
# Create figure
def display_wordcloud(width=400, height=200, scl_factor=0.5, img_path='', display_flag=False):
    """
        Original code can be seen at Plotly site.  Modified so that it takes in an image file
        For
    """
    fig = go.Figure()

    # Constants
    img_width = width
    img_height = height
    scale_factor = scl_factor

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(
            x=[0, img_width * scale_factor],
            y=[0, img_height * scale_factor],
            mode="markers",
            marker_opacity=0
        )
    )

    # Configure axes
    fig.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )

    fig.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )

    # Add image
    fig.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=img_path)
    )

    # Configure other layout
    fig.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )
    
    if display_flag:
        # Disable the autosize on double click because it adds unwanted margins around the image
        # More detail: https://plot.ly/python/configuration-options/
        fig.show(config={'doubleClick': 'reset'})
    
    return fig
```

In this block, we create the word cloud for the specified verison and book.


```python
stop_list = get_stop_words(file_path='./assets/resources/custom_stopwords.txt')
image_width, image_height = 800, 800
```

Uncoment the lines below to and change the version & book fields, to generate plot.ly figure for respective version and book


```python
# df_book = df.loc[((df.version=='asv') & (df.book=='john'))]

# image = generate_word_cloud(df_book, col='text', stopwords=stop_list)
# fig = display_wordcloud(width=image_width, height=image_height, img_path=image, display_flag=True)
```


```python
# df_book = df.loc[((df.version=='ylt') & (df.book=='john'))]

# image = generate_word_cloud(df_book, col='text', stopwords=stop_list)
# fig = display_wordcloud(width=image_width, height=image_height, img_path=image, display_flag=True)
```

_Grid of WordClouds_


```python
def generate_multi_wordcloud(df=None, version_list=None, book='genesis', display_flag=False,
                            stopword_list=None, grid_size=(33., 33.), image_width=400, 
                            image_height=400, rows=4, cols=3, axs_pad=0.5):
    
    fig = plt.figure(figsize=grid_size)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(rows, cols),  # creates rows x cols grid of axes
                     axes_pad=axs_pad,  # pad between axes in inch.
                     )
    
    for ax, version in zip(grid, version_list):
        
        df_book = df.loc[((df.version==version) & (df.book==book))]

        image = generate_word_cloud(df_book, col='text', stopwords=stopword_list, 
                                    img_height=image_height, img_width=image_width)        
        ax.imshow(image)
        ax.set_title(version.upper())
                
    if display_flag:
        plt.show()
    
    return fig
```


```python
def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    site: http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
    """
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf
```


```python
def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    site: http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )
```


```python
def get_img_from_fig(fig, dpi=180):
    """
        Function that generates high definition image and returns it as a numpy array
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
```


```python
cloud_grid = generate_multi_wordcloud(df=df, version_list=df.version.unique(), display_flag=True,
                                     book='isaiah', stopword_list=stop_list)
```


![png](Bible_Comparison_files/Bible_Comparison_95_0.png)



```python
img = get_img_from_fig(cloud_grid)
pillow_image = Image.fromarray(img)
pillow_image.show()
```


```python
# display_wordcloud(img_path=pillow_image, width=1300, height=1300)
```

# Conclusion

At the end, we found that the versions are similar to one another, with some differences that arise.

__Metrics__ <br>

The metrics gave us a superficial means of comparing different versions and seeing how they relate and differ. There seemed to be more similarities overall than differences.  The 4 general statistics used to quantify and visualize the differences were *sum, mean, median,* and _mode._  As we looked at the sums, the character counts were almost identical across all versions of the Bibles compared.  

The main metrics that saw the most drastic differences from version to version were **punctuation_count, title_word_count,** and **upper_case_word_count.**  

In the graph below, the upper case word count is significantly higher for _AKJV, KJV, KJV1900, and WB_ versions.  Versions of the KJV are noted to be more poetic in interpretation, as well as in the printing of the verses. This same philosophy
It is possible that the other versions happen to be lower due to new scripts and advancements in translations that made prior readings more clear and understandable.  This difference in the upper case count also points towards a shift in pronouns and titles used to identify GOD vs those used to identify man.

![Average upper_case_word_count_by_verse](./assets/resources/images/average_upper_case_word_count_by_verse.png)


To better view the differences in the metrics, I highly recommend running the Dashboard  app.py file, contained within the repository.  Also looking further into the philosophies behind each translation may answer some of the trends visible within the metrics.

__Word2Vec__ <br>

Modeling in Word2Vec was very interesting.  Many of the versions returned at least 2 shared words when we sought to look at words that were most similar.  The results seemed strange once we decided to look at least similar words.  More research would have to be done to better understand what attributed to the difference in results seen.

__Word cloud representation__ <br>

I found it interesting comparing the word cloud generations to one another vs the Word2Vec representations.  In the multi-grid wordclouds, many of the same words appeared accross each available version.  The main difference was the font size.  The wordcloud package in use utlizes word frequency to set the font size of each respective word.  In the grid above, we are looking at the book of Isaiah.  All versions contains the words LORD, but it shows up with different sizes.  For other versions such as the WEB, the personal name of GOD is writtten into the translation, in place of the all caps "LORD."  Many of the older versions carry this practice. Based on the results of the word clouds, we can see that there is still a great deal of commonality among the different versions, despite how language has evolved overtime and how translation techniques and paradigms have grown as well.

At the end of the day, a deeper comparison can be made, but it will take much more human intervention to see the qualitive differences.  This remark does not discourage the use of technology to help observe difference in versions. There is still so much more that we can create code to identify.  This is very exciting to me because there is so much more room in understanding Biblical text and developign translations that will better communicate the Bible's central theme to new generations. Data Science can play a very substantial role in bridging the communication gap.

# Future Works & Additions

* Add a customizable lematizer to take care of different forms of the same word.
  Ex: said, saith, came, come, cometh
* Develop functionality for the Dash app that will generate the different word cloud images and pass them through a CNN to find which ones are the most similar
* Add functionality that would use the Word2Vec to determine uniqueness vs similarities of the vectors returned.
* Make the wordcloud grid available in the Dash app, where each grid would be a subplot that users can interact with
* Add similar word functionality to the Dash to allow users to see the output of the Word2Vec models in real-time
* Extend the project to automatically compare and contrast other documents
* Using a CNN on the different word clouds generated and trainning it to identify versions that are closest, based on the text passed in and|or the word cloud images generated. 
