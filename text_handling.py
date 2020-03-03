import re
import string
import nltk
from wordcloud import STOPWORDS, WordCloud
from nltk import word_tokenize 
from nltk.corpus import stopwords


def remove_nonalnum_lead_trail(s):
    try:
        if not s.isalnum():
            if not s[-1].isalnum():
                s = s.strip(s[-1])
            if not s[0].isalnum():
                s = s.strip(s[0])
        return s
    except:
#         print(f"error has occurred.  Word: {s}")
        pass

def remove_punct(word_list=None):
    """
        Function to remove punctuations characters from list of words passed in.
    """
    updated_word_list = []
    
    additional_chars = list('､—’')
    
    punctuation_list = list(string.punctuation)
    punctuation_list += additional_chars
    
#     print(punctuation_list)
    
    if word_list:
#         updated_words = [remove_nonalnum_lead_trail(item) for item in list_of_words if item not in punctuation_list ]  
        for item in word_list:
            cleaned_word = remove_nonalnum_lead_trail(item)
            if item not in punctuation_list and cleaned_word != None:
                updated_word_list.append(cleaned_word)
                
    return updated_word_list

def get_stop_words(file_path=''):
    
    complete_stoplist = list(STOPWORDS) + list(stopwords.words('english'))
    
    try:
        if file_path:
            with open(file_path, 'r') as f:
                eliz_stopwords = f.readlines()
                
            eliz_stopwords = [word.strip() for word in eliz_stopwords]
            complete_stoplist += eliz_stopwords
        
        return set(complete_stoplist)

    except:
        print('#######################')
        print('Issue reading the file.')
        print('#######################')
        
        return set(complete_stoplist)


def remove_stop_words(list_of_words=None):
    pass
    
def parse_book(book_dict=None, tokenize_flag=False):
    b_name = book_dict['book_name']
    chapters = book_dict['book'].keys() #get chapter numbers
    parsed_book = {}
    
    #Loop through chapter numbers and get verse content
    for c_num in chapters:        
        chapter = book_dict['book'][c_num]['chapter']
        verses = list(chapter.keys())
        
        #Loop through verses & parse content
        for v_num in verses:
            verse = chapter[v_num]['verse'].strip()
            if tokenize_flag:
                tokenized_verse = remove_punct(word_tokenize(verse))
                parsed_book[f'{b_name} {c_num}:{v_num}'] = tokenized_verse            
            else:
                parsed_book[f'{b_name} {c_num}:{v_num}'] = verse
    return parsed_book

def parse_formatted_verse_ref(text='', tokenize_flag=False):
    """
        This method takes in fully formatted bible verse text and stores it in a dictionary.
        Passing text that is too large into this function may cause issues within Jupyter notebook.  Be advised
            
        Argument:
            
            text {str}:
                Formatted Bible text sea
        
        Return:
            parsed_dict {dict}:
                Dictionary containing parsed verses with the information in the following format
                    
                    key {str}: {Book}{chapter}:{verse}
                    value {ist}: tokenized bible verse text

    """
    search = re.findall(r'^(.*:\d+)(.*)', text.strip(), re.M|re.I)
    if tokenize_flag:
        search_dict = {item[0]: tokenize_verse(item[1].strip()) for item in search}
    else:
        search_dict = {item[0]: item[1].strip() for item in search}
    return search_dict

def tokenize_verse(verse):
    """
        Function to take in a verse and tokenize the verse. Punctuations at the beggining and end of words will be stripped
    """
    list_of_words = word_tokenize(verse.strip())
    tokenized_verse = remove_punct(list_of_words)
    return tokenized_verse

def generate_word_cloud(df=None, col='', stopwords=None,  img_width=800, img_height=800):
    """
        This function takes text and turns it into a word cloud.
        Returns a Pillow image representation of the word cloud
    """
    comment_words = ' '

    for val in df[col]:
        tokens = [token for token in tokenize_verse(val)]

        for words in tokens:
            comment_words = comment_words + words + ' '

    wordcloud = WordCloud(width = img_width, height = img_height, 
                    background_color ='white',
                    stopwords = stopwords, 
                    min_font_size = 10).generate(comment_words)

    return wordcloud.to_image()