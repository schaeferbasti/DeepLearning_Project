# import packages
from datasets import load_dataset
import pandas as pd
import nltk
import pickle
from nltk.stem.snowball import FrenchStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



string.punctuation

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# load data
def load_data():
        """
        load data
        """
        return load_dataset('Nicolas-BZRD/Parallel_Global_Voices_English_French',
                                    split='train').to_pandas()


def remove_punctuation(text):
        """
        defining the function to remove punctuation
        """
        punctuation_free="".join([i for i in text if i not in string.punctuation])
        return punctuation_free


def save_punct_free(df:pd.DataFrame):
        """
        storing the puntuation free text
        Args:
            df - df to update
        """
        df['en'] = df['en'].apply(lambda x:remove_punctuation(x))
        df['fr'] = df['fr'].apply(lambda x:remove_punctuation(x))
        return df


def lower_case(df):
        """
        change text to lower case
        Args:
            df - df to update
        """  
        df['en'] = df['en'].apply(lambda x: x.lower())
        df['fr'] = df['fr'].apply(lambda x: x.lower())
        return df


def tokenize_data(df):
        """
        tokenize text
        Args:
            df - df to update
        """  
        df['en'] = df['en'].apply(lambda row: nltk.word_tokenize(row[0]))#, axis=1)
        df['fr'] = df['fr'].apply(lambda row: nltk.word_tokenize(row[1]))#, axis=1)
        return df
  

def get_word_index(column):  
        """
        Args: column - column to get word index on for sequencing
        """
        # name out of vocabulary character and tokenize
        tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
        
        # fit tokenizer on text
        tokenizer.fit_on_texts(column.tolist())

        # convert words to  index
        word_to_index = tokenizer.word_index

        # convert index to word
        index_to_word = {idx: word for word, idx in word_to_index.items()}

        # create sequences with word indexes
        sequences = tokenizer.texts_to_sequences(column.tolist())
        # print(f'Word index: {word_to_index}')
        # print(f'\nSequences: {sequences}')

        return word_to_index, index_to_word, sequences


def add_padding(column, sequences):
        '''
        Args: column - column to be padded
              sequences - encoded word sequence
        '''
        lengths_of_rows = len(column.tolist())
        padded_seqs = pad_sequences(sequences, maxlen=lengths_of_rows, padding='post',)
        # print(f'Padded sequences: {padded_seqs}')
        return padded_seqs, lengths_of_rows
