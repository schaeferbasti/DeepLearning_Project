import pandas as pd
import numpy as np
from datasets import load_dataset

import nltk
from cleantext import clean
from nltk.tokenize import word_tokenize

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Embedding, SimpleRNN, Dense, GRU, LSTM, Bidirectional, Dropout, Input, Dense
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

# Extract dataset
translation_dataset = load_dataset('Nicolas-BZRD/Parallel_Global_Voices_English_French',
                                   split='train').to_pandas()
translation_dataset.head(5)

df = translation_dataset.copy()

# First step - Data Pre-processing

# nltk downloads
nltk.download('punkt')

# Define a cleaning function
def clean_text(text):
    return clean(text,
                 fix_unicode=True,               # fix various unicode errors
                 to_ascii=True,                  # transliterate to closest ASCII representation
                 lower=True,                     # lowercase text
                 no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
                 no_urls=True,                   # replace all URLs with a special token
                 no_emails=True,                 # replace all email addresses with a special token
                 no_phone_numbers=True,          # replace all phone numbers with a special token
                 no_numbers=False,               # replace all numbers with a special token
                 no_digits=False,                # replace all digits with a special token
                 no_currency_symbols=True,       # replace all currency symbols with a special token
                 no_punct=True,                  # remove punctuations
                 replace_with_punct="",          # replace punctuations with this character
                 replace_with_url="<URL>",
                 replace_with_email="<EMAIL>",
                 replace_with_phone_number="<PHONE>",
                 replace_with_number="<NUMBER>",
                 replace_with_digit="<DIGIT>",
                 replace_with_currency_symbol="<CUR>",
                 lang="en")

# Apply cleaning function to both English and French columns
df['en'] = df['en'].apply(clean_text)
df['fr'] = df['fr'].apply(clean_text)

# Tokenization
df['en_tokens'] = df['en'].apply(word_tokenize)
df['fr_tokens'] = df['fr'].apply(word_tokenize)

# Handling missing data
df.dropna(subset=['en', 'fr'], inplace=True)

# Save the preprocessed data
df.to_csv('preprocessed_data.csv', index=False)