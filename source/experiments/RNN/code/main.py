# --- 1. We import the libraries we need ---
import numpy as np
import tensorflow as tf
import pandas as pd
import time
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, Callback
from method.simple_rnn import SimpleRNNModel
from method.simple_lstm import SimpleLSTMModel
from method.simple_gru import SimpleGRUModel
from method.bi_rnn import BidirectionalRNNModel
from method.bi_lstm import BidirectionalLSTMModel
from method.bi_gru import BidirectionalGRUModel
from method.auto_rnn import EncoderDecoderRNNModel
from method.auto_lstm import EncoderDecoderLSTMModel
from method.auto_gru import EncoderDecoderGRUModel

# --- 2. We definie testing modules ---

def translate_sequence(seq, tokenizer):
    """ Translates a sequence of integers back into text using the tokenizer. """
    words = [tokenizer.index_word.get(idx, '') for idx in seq]
    return ' '.join(words).strip()

def predict_and_compare(index, testX, model, tokenizer_en, tokenizer_fr):
    """ Predicts translation for a given index in the test set and compares with the ground truth. """
    input_seq = testX[index:index+1]
    prediction = model.predict(input_seq)

    # Converting the prediction to a sequence of integers
    predicted_seq = np.argmax(prediction, axis=-1)[0]

    # Reverse tokenization (converting sequences back to words)
    input_text = translate_sequence(input_seq[0], tokenizer_en)
    predicted_text = translate_sequence(predicted_seq, tokenizer_fr)
    ground_truth_text = translate_sequence(testY[index].flatten(), tokenizer_fr)

    # Return results
    return input_text, predicted_text, ground_truth_text

def predict_and_compare_auto_en(index, testX, testY, model, tokenizer_en, tokenizer_fr):
    """ Predicts translation for a given index in the test set and compares with the ground truth. """
    input_seq_X = testX[index:index+1]
    input_seq_Y = testY[index:index+1]
    prediction = model.predict([input_seq_X, input_seq_Y])

    # Converting the prediction to a sequence of integers
    predicted_seq = np.argmax(prediction, axis=-1)[0]

    # Reverse tokenization (converting sequences back to words)
    input_text = translate_sequence(input_seq_X[0], tokenizer_en)
    predicted_text = translate_sequence(predicted_seq, tokenizer_fr)
    ground_truth_text = translate_sequence(testY[index].flatten(), tokenizer_fr)

    # Return results
    return input_text, predicted_text, ground_truth_text


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

# --- 3. We check the gpus available ---
    
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
    
    # --- 4. We define global variables ---
    
    EPOCHS = 100
    BATCH_SIZE = 32
    MAX_VOCAB_SIZE_FR = 20500

    # --- 3. We open the data and apply tokenization ---

    df = pd.read_csv('preprocessed_data.csv')

    tokenizer_en = Tokenizer()
    tokenizer_en.fit_on_texts(df['en_tokens'])
    tokenizer_fr = Tokenizer(num_words=MAX_VOCAB_SIZE_FR + 1)
    tokenizer_fr.fit_on_texts(df['fr_tokens'])

    # Convert text to sequences
    sequences_en = tokenizer_en.texts_to_sequences(df['en_tokens'])
    sequences_fr = tokenizer_fr.texts_to_sequences(df['fr_tokens'])

    # Padding sequences
    max_len = max(max(len(s) for s in sequences_en), max(len(s) for s in sequences_fr))
    sequences_en = pad_sequences(sequences_en, maxlen=max_len, padding='post')
    sequences_fr = pad_sequences(sequences_fr, maxlen=max_len, padding='post')

    # Splitting the data
    split = int(len(sequences_en) * 0.8)
    trainX, testX = sequences_en[:split], sequences_en[split:]
    trainY, testY = sequences_fr[:split], sequences_fr[split:]

    # Finally, reshape data for feeding into model (French words)
    trainY = trainY.reshape(trainY.shape[0], trainY.shape[1], 1)
    testY = testY.reshape(testY.shape[0], testY.shape[1], 1)

    # --- 4. We load the model ---

    #method_name = ['SimpleRNN', 'SimpleLSTM', 'SimpleGRU', 'BiRNN', 'BiLSTM', 'BiGRU', 'EncoderDecoderRNN', 'EncoderDecoderLSTM', 'EncoderDecoderGRU']
    method_name = ['EncoderDecoderRNN', 'EncoderDecoderLSTM', 'EncoderDecoderGRU']
    #method_instance = [SimpleRNNModel(tokenizer_en, tokenizer_fr, max_len, MAX_VOCAB_SIZE_FR), SimpleLSTMModel(tokenizer_en, tokenizer_fr, max_len, MAX_VOCAB_SIZE_FR), SimpleGRUModel(tokenizer_en, tokenizer_fr, max_len, MAX_VOCAB_SIZE_FR), BidirectionalRNNModel(tokenizer_en, tokenizer_fr, max_len, MAX_VOCAB_SIZE_FR), BidirectionalLSTMModel(tokenizer_en, tokenizer_fr, max_len, MAX_VOCAB_SIZE_FR), BidirectionalGRUModel(tokenizer_en, tokenizer_fr, max_len, MAX_VOCAB_SIZE_FR), EncoderDecoderRNNModel(tokenizer_en, tokenizer_fr, MAX_VOCAB_SIZE_FR), EncoderDecoderLSTMModel(tokenizer_en, tokenizer_fr, MAX_VOCAB_SIZE_FR), EncoderDecoderGRUModel(tokenizer_en, tokenizer_fr, MAX_VOCAB_SIZE_FR)]
    method_instance = [EncoderDecoderRNNModel(tokenizer_en, tokenizer_fr, MAX_VOCAB_SIZE_FR), EncoderDecoderLSTMModel(tokenizer_en, tokenizer_fr, MAX_VOCAB_SIZE_FR), EncoderDecoderGRUModel(tokenizer_en, tokenizer_fr, MAX_VOCAB_SIZE_FR)]
   

    # Shared Callbacks
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=1)

    for i in range(len(method_name)):
        current_model = method_instance[i].build_model()

        # --- 5. We train the model ---
        checkpoint = ModelCheckpoint(f'./results/weights/weights_{method_name[i]}.best.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        csv_logger = TimedCSVLogger(f'./results/training_log/training_log_{method_name[i]}.csv', append=True)

        if method_name[i] == 'EncoderDecoderRNN' or method_name[i] == 'EncoderDecoderLSTM' or method_name[i] == 'EncoderDecoderGRU':
            current_model.fit([trainX, np.squeeze(trainY, axis=-1)], trainY, epochs=EPOCHS, validation_split=0.2, batch_size=BATCH_SIZE, callbacks=[checkpoint, csv_logger, early_stopping])
        else:
            current_model.fit(trainX, trainY, epochs=EPOCHS, validation_data=(testX, testY), batch_size=BATCH_SIZE, callbacks=[checkpoint, csv_logger, early_stopping])

        # --- 6. We test the model (Change for more meaningful metrics like BLEU) ---

        all_predictions = []
        for j in range(5):
            if method_name[i] == 'EncoderDecoderRNN' or method_name[i] == 'EncoderDecoderLSTM' or method_name[i] == 'EncoderDecoderGRU':
                input_text, predicted_text, ground_truth_text = predict_and_compare_auto_en(index=j, testX=testX, testY=np.squeeze(testY, axis=-1), model=current_model, tokenizer_en=tokenizer_en, tokenizer_fr=tokenizer_fr)
            else:
                input_text, predicted_text, ground_truth_text = predict_and_compare(index=j, testX=testX, model=current_model, tokenizer_en=tokenizer_en, tokenizer_fr=tokenizer_fr)
            all_predictions.append((input_text, predicted_text, ground_truth_text))

        # Writing predictions to a text file
        with open(f'./results/predictions/model_predictions_{method_name[i]}.txt', 'w', encoding='utf-8') as file:
            for input_text, predicted_text, ground_truth in all_predictions:
                file.write("Input (English): " + input_text + "\n")
                file.write("Predicted (French): " + predicted_text + "\n")
                file.write("Ground Truth (French): " + ground_truth + "\n")
                file.write("----------\n")
