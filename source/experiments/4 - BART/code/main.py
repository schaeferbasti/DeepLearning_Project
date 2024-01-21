# --- 1. We import the libraries we need ---
import numpy as np
import tensorflow as tf
import pandas as pd
import time
from transformers import TFMarianMTModel, MarianTokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from jiwer import wer
from bert_score import score as bert_score
import pyter
import re
import ast
import os

def predict_and_compare(index, testX, testY, model, tokenizer, max_output_length=5):
    """ Predicts translation for a given index in the test set and compares with the ground truth. """
    input_seq = testX[index:index+1]

    # Determine the total max_length (input length + desired output length)
    total_max_length = len(input_seq[0]) + max_output_length
    prediction = model.generate(input_seq, max_length=total_max_length, no_repeat_ngram_size=2)

    # Decode the prediction and input
    input_text = tokenizer.decode(input_seq[0], skip_special_tokens=True)
    predicted_text = tokenizer.decode(prediction[0], skip_special_tokens=True)

    # For ground truth
    ground_truth_text = tokenizer.decode(testY[index], skip_special_tokens=True)

    # Return results
    print(f'Prediction index for element finished: {index}')
    return input_text, predicted_text, ground_truth_text

def clean_predictions(predictions):
    """ Cleans predictions for comparison purposes. """
    inputs, preds, truths = zip(*predictions)

    # Join the tokenized words into sentences
    preds = [" ".join(pred) if isinstance(pred, list) else pred for pred in preds]
    truths = [" ".join(truth) if isinstance(truth, list) else truth for truth in truths]

    # Clean preds
    cleaned_preds = []
    for current_pred in preds:
      tokens = current_pred.strip('[]').split(',')

      # Cleaning each token by removing special characters and extra quotes
      cleaned_tokens = [re.sub(r'[^\w\s]', '', token.strip()) for token in tokens]
      split_tokens = [word for token in cleaned_tokens for word in token.split()]
      cleaned_preds.append(split_tokens)

    return zip(inputs, cleaned_preds, truths)

class TimedCSVLogger(CSVLogger):
    def __init__(self, filename, separator=',', append=False):
        super().__init__(filename, separator, append)
        self.start_time = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        end_time = time.time()
        logs['epoch_duration'] = end_time - self.epoch_start_time
        logs['total_time'] = end_time - self.start_time
        super().on_epoch_end(epoch, logs)

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

    BATCH_SIZE = 32
    EPOCHS = 500  #It took in total 392 to fine-tune it!
    VALIDATION_SPLIT = 0.2
    identifier = 'marian_transformer_mt'

    # --- 3. We open the data and apply tokenization, with data generator ---

    df = pd.read_csv('./drive/MyDrive/data/dl/preprocessed_data.csv')
    tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-fr')

    # We extract the test set first
    train_df, test_df = train_test_split(df, test_size=VALIDATION_SPLIT)
    testX = tokenizer(test_df['en_tokens'].tolist(), return_tensors='tf', padding=True, truncation=True, max_length=512)['input_ids']
    testY = tokenizer(test_df['fr_tokens'].tolist(), return_tensors='tf', padding=True, truncation=True, max_length=512)['input_ids']


    src_texts = train_df['en_tokens'].tolist()
    tgt_texts = train_df['fr_tokens'].tolist()

    model_inputs = tokenizer(src_texts, return_tensors='tf', padding=True, truncation=True, max_length=512)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(tgt_texts, return_tensors='tf', padding=True, truncation=True, max_length=512)

    model_inputs["labels"] = labels["input_ids"]

    # Prepare decoder_input_ids
    start_token_id = tokenizer.pad_token_id
    decoder_input_ids = np.full_like(labels['input_ids'], start_token_id)
    decoder_input_ids[:, 1:] = labels['input_ids'][:,:-1]

    model_inputs["decoder_input_ids"] = decoder_input_ids

    def data_generator(model_inputs, batch_size):
        total_size = len(model_inputs['input_ids'])
        for i in range(0, total_size, batch_size):
            batch_input_ids = model_inputs['input_ids'][i:i + batch_size]
            batch_attention_mask = model_inputs['attention_mask'][i:i + batch_size]
            batch_decoder_input_ids = model_inputs['decoder_input_ids'][i:i + batch_size]
            batch_labels = labels['input_ids'][i:i + batch_size]
            batch_decoder_input_ids = model_inputs['decoder_input_ids'][i:i + batch_size]

        yield ({"input_ids": batch_input_ids, "attention_mask": batch_attention_mask, "decoder_input_ids": batch_decoder_input_ids}, batch_labels)


    # Split data into training and validation
    train_size = int((1 - VALIDATION_SPLIT) * len(model_inputs['input_ids']))
    train_dataset = (model_inputs[:train_size], labels[:train_size])
    validation_dataset = (model_inputs[train_size:], labels[train_size:])

    # Convert dataset and charg into model
    train_data = tf.data.Dataset.from_generator(
    lambda: data_generator(model_inputs, BATCH_SIZE),
    output_types=({'input_ids': tf.int32, 'attention_mask': tf.int32, 'decoder_input_ids': tf.int32}, tf.int32),
    output_shapes=({'input_ids': tf.TensorShape([None, None]), 'attention_mask': tf.TensorShape([None, None]), 'decoder_input_ids': tf.TensorShape([None, None])},
                   tf.TensorShape([None, None]))
    ).prefetch(tf.data.experimental.AUTOTUNE)

    validation_data = tf.data.Dataset.from_generator(
    lambda: data_generator(model_inputs, BATCH_SIZE),
    output_types=({'input_ids': tf.int32, 'attention_mask': tf.int32, 'decoder_input_ids': tf.int32}, tf.int32),
    output_shapes=({'input_ids': tf.TensorShape([None, None]), 'attention_mask': tf.TensorShape([None, None]), 'decoder_input_ids': tf.TensorShape([None, None])},
                   tf.TensorShape([None, None]))
    ).prefetch(tf.data.experimental.AUTOTUNE)

    # --- 4.1 Define Callbacks ---
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)
    csv_logger = TimedCSVLogger(f'./drive/MyDrive/data/dl/results/training_log/training_log_{identifier}.csv', append=True)

    # --- 4.2 We define and compile the model ---
    model = TFMarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-fr')

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss)

    model_weights_path = f'./drive/MyDrive/data/dl/results/weights/weights_{identifier}.best'

    if os.path.exists(model_weights_path):
      model = tf.keras.models.load_model(model_weights_path)
      print("Model weights loaded successfully!")

    # --- 5. We train the model ---
    model.fit(train_data, validation_data=validation_data, epochs=EPOCHS, callbacks=[early_stopping, csv_logger])
    model.save_weights(model_weights_path)

    # ---6. Measure the performance ---
    all_predictions_raw = []
    for j in range(100):
        input_text, predicted_text_raw, ground_truth_text = predict_and_compare(j, testX, testY, model, tokenizer)
        all_predictions_raw.append((input_text, predicted_text_raw, ground_truth_text))
    all_predictions = clean_predictions(all_predictions_raw)

    with open(f'./drive/MyDrive/data/dl/results/predictions/model_predictions_{identifier}.txt', 'w', encoding='utf-8') as file:
      for input_text, predicted_text, ground_truth in all_predictions:
          # Format the input_text list
          input_text = ast.literal_eval(input_text)
          ground_truth = ast.literal_eval(ground_truth)

          formatted_input_text = "Input (English): " + " ".join(f"'{word}'" for word in input_text)
          formatted_pred_text = "Predicted (French): " + " ".join(f"'{word}'" for word in predicted_text)
          formatted_truth_text = "Ground Truth (French): " + " ".join(f"'{word}'" for word in ground_truth)

          file.write(formatted_input_text + "\n")
          file.write(formatted_pred_text + "\n")
          file.write(formatted_truth_text + "\n")
          file.write("----------\n")
