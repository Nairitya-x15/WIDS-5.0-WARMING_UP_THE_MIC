import pandas as pd
import re
import string
import joblib

user_review = input("Enter the Text for which you want to test sentiment: ")

model = joblib.load("WEEK 1/sentiment_model.pkl")
vectorizer = joblib.load("WEEK 1/tfidf_vectorizer.pkl")

def remove_html_tags(text):
    pattern= re.compile('<.*?>')
    return pattern.sub(r'', text)

def remove_punctuation_translate(input_string):
    translator= str.maketrans('','',string.punctuation)
    cleaned_string= input_string.translate(translator)
    return cleaned_string

def preprocess_text(text):
    text = text.lower()
    text = remove_html_tags(text)
    text = remove_punctuation_translate(text)
    return text

user_review = preprocess_text(user_review)
user_review_tfidf = vectorizer.transform([user_review])
prediction = model.predict(user_review_tfidf)

print("Predicted Sentiment:", "Positive" if prediction[0] == 1 else "Negative")
