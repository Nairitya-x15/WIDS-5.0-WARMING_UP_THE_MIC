print("THIS PROGRAM DEMONSTRATES TOKENIZATION AND STEMMING ON IMDB DATASET")
print("Please wait it may take 1-2 minutes.......")

import pandas as pd
import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

df= pd.read_csv('WEEK 1/IMDB-Dataset.csv')

df["review"]= df["review"].str.lower();

def remove_html_tags(text):
    pattern= re.compile('<.*?>')
    return pattern.sub(r'', text)
df["review"]= df['review'].apply(remove_html_tags)

def remove_punctuation_translate(input_string):
    translator= str.maketrans('','',string.punctuation)
    cleaned_string= input_string.translate(translator)
    return cleaned_string
df["review"]= df['review'].apply(remove_punctuation_translate)

df["tokens"] = df["review"].apply(word_tokenize)

ps= PorterStemmer()
def stemming(text):
   stemmed_words = [ps.stem(word) for word in text]
   return stemmed_words;
df["stemmed_tokens"]=df["tokens"].apply(stemming)

df["unique_tokens"] = df["stemmed_tokens"].apply( lambda x: list(dict.fromkeys(x)) )

print("IMDB DATASET AFTER TEXT CLEANING, TOKENIZATION AND STEMMING:")
print(df["unique_tokens"])