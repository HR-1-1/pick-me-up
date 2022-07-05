import fasttext
import pandas as pd
import numpy as np
import string
import gensim
import spacy
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import tokenize
from nltk.stem import WordNetLemmatizer
import re

# Preprocessing functions
def normalize_text (text):
    text = text.str.lower() # lowercase
    text = text.str.replace(r"\#"," ", regex=True) # replaces hashtags
    text = text.str.replace(r"http\S+","URL", regex=True)  # remove URL addresses
    text = text.str.replace(r"@"," ", regex=True)
    text = text.str.replace("\s{2,}", " ", regex=True)
    return text

def master_pre_process(df):
  df.text = normalize_text(df.text)
  regex = re.compile('[%s]' % re.escape(string.punctuation))
  regex1 = re.compile('[%s]' % re.escape(string.digits))
  df.text = df.text.apply(lambda x: regex.sub(' ',x))
  df.text = df.text.apply(lambda x: regex1.sub(' ',x))
  return df

#Extract the zip-folder named "Submission" and place it in a folder and give the path to the folder below
path = 'drive/MyDrive/ML/DL_Hackathon/'

test = pd.read_csv(path + 'Submission/test.csv')
submission = pd.read_csv(path + 'Submission/sample_submission.csv')
model = fasttext.load_model(path + 'Submission/ProductCategorizationModel.bin')

print("Code for Inference")
test['text'] = test.title + " " + test.content
test.drop(labels=['uid','title','content'], axis=1, inplace=True)
test = master_pre_process1(test)
submission.target_ind = test.text.apply(lambda x: int(model.predict(x)[0][0].split('__')[2]))

# Save Submission File
submission.to_csv(path + 'Submission/submission.csv', index = False) 
