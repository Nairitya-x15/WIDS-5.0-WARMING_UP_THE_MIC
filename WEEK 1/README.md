# WIDS 5.0 -- WEEK 1 ASSIGNMENT
### TASK 1: Implement tokenization & stemming using Python + NLTK / spaCy
-> I Have used IMDB Dataset to implement Tokenization and stemming.
<br>
-> Firstly read the IMDB-Dataset.csv file and import important packages like pandas, string, re, nltk.
<br>
-> Clear the dataset my removing html tags, punctuation mark.
<br>
-> Then perform tokenization and stemming using nltk library.
<br>
### TASK 2: Build a TF-IDF text classifier for sentiment (IMDb or SST-2)
->This task has been done in 2 parts.
<br>
#### 1) SENTIMENT_MODEL TRAINING :--
-> Firstly import important packages and library
<br>
-> Read the IMDB-Dataset.csv
<br>
-> Clean dataset by removing html tags and punctuation marks.
<br>
-> Split the dataset into 4:1 ratio in training set and testing dataset respectively.
<br>
-> Perform TF-IDF vectorization using sklearn library
<br>
-> Use logistic regression for binary sentiment classification
<br>
-> Save the model using joblib
#### 1) SENTIMENT PREDICTION :--
-> Load the sentiment_model using joblib
<br>
->Take input text from user
<br>
-> Perform basic text cleaning on input text like html tag removal, remove punctuation mark.
<br>
->And use sentiment model to predict sentiment of input text
### TASK 3: Explore Hugging Face datasets
-> I explored https://huggingface.co to gain some knowledge about dataset and i also found IMDB-Dataset, SST-2 Dataset there.

# END OF WEEK 1



