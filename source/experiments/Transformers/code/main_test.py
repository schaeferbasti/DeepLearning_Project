from datasets import load_dataset
import nltk
from cleantext import clean
from nltk.tokenize import word_tokenize
import numpy as np
import tensorflow as tf
import pandas as pd
import time
from transformers import TFMarianMTModel, MarianTokenizer

def prepro():
    # Extract dataset
    translation_dataset = load_dataset('Nicolas-BZRD/Parallel_Global_Voices_English_French',
                                    split='train').to_pandas()

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


if __name__ == '__main__':

    # Call Pre-processing:
    prepro()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Set TensorFlow to use only one GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

            # Enable memory growth
            tf.config.experimental.set_memory_growth(gpus[0], True)

            print("Using GPU:", gpus[0])
        except RuntimeError as e:
            # Memory growth must be set at program startup
            print("RuntimeError:", e)
    else:
        raise SystemError("GPU device not found")
    
    # --- 2. We define the global variable ---

    BATCH_SIZE = 8
    EPOCHS = 100

    # --- 3. We open the data and apply tokenization, with data generator ---

    df = pd.read_csv('preprocessed_data.csv')
    tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
    src_texts = df['en_tokens'].tolist()
    tgt_texts = df['fr_tokens'].tolist()

    model_inputs = tokenizer(src_texts, return_tensors='tf', padding=True, truncation=True, max_length=512)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(tgt_texts, return_tensors='tf', padding=True, truncation=True, max_length=512)

    model_inputs["labels"] = labels["input_ids"]

    def data_generator(model_inputs, labels, batch_size):
        total_size = len(model_inputs['input_ids'])
        for i in range(0, total_size, batch_size):
            batch_input_ids = model_inputs['input_ids'][i:i + batch_size]
            batch_attention_mask = model_inputs['attention_mask'][i:i + batch_size]
            batch_labels = labels['input_ids'][i:i + batch_size]

            batch = {
                'input_ids': batch_input_ids, 
                'attention_mask': batch_attention_mask, 
                'labels': batch_labels
            }
            yield batch

    # --- 4. We define the model ---
    model = TFMarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-fr')

    # --- 5. We perform translation and decode the output ---

    for batch in data_generator(model_inputs, labels, BATCH_SIZE):
        translated = model.generate(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask']
        )
        translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        print(translated_text)

