# --- 1. We import the libraries we need ---
import numpy as np
import tensorflow as tf
import pandas as pd
import time
from transformers import TFMarianMTModel, MarianTokenizer

if __name__ == '__main__':

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



