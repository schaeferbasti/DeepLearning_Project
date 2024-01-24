# --- 1. We import the libraries we need ---
import numpy as np
import tensorflow as tf
import pandas as pd
import time
import os
import torch
import ast

from contextlib import redirect_stdout

from torchmetrics.text.wer import WordErrorRate
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text import TranslationEditRate
from torchmetrics.text.bert import BERTScore
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, Callback

from method.SAN_CNN_Attention import SAN_CNN_Attention
from method.SAN_CNN_MultiHeadAttention import SAN_CNN_MultiHeadAttention
from method.SAN_GRU_Attention import SAN_GRU_Attention
from method.SAN_GRU_MultiHeadAttention import SAN_GRU_MultiHeadAttention
from method.SAN_LSTM_Attention import SAN_LSTM_Attention
from method.SAN_LSTM_MultiHeadAttention import SAN_LSTM_MultiHeadAttention
from method.SAN_RNN_Attention import SAN_RNN_Attention
from method.SAN_RNN_MultiHeadAttention import SAN_RNN_MultiHeadAttention


# --- 2. We define testing modules ---

def translate_sequence(seq, tokenizer):
    """ Translates a sequence of integers back into text using the tokenizer. """
    words = [tokenizer.index_word.get(idx, '') for idx in seq]
    return ' '.join(words).strip()


def predict_and_compare(index, testX, model, tokenizer_en, tokenizer_fr):
    """ Predicts translation for a given index in the test set and compares with the ground truth. """
    input_seq = testX[index:index + 1]
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
    input_seq_X = testX[index:index + 1]
    input_seq_Y = testY[index:index + 1]
    prediction = model.predict([input_seq_X, input_seq_Y])

    # Converting the prediction to a sequence of integers
    predicted_seq = np.argmax(prediction, axis=-1)[0]

    # Reverse tokenization (converting sequences back to words)
    input_text = translate_sequence(input_seq_X[0], tokenizer_en)
    predicted_text = translate_sequence(predicted_seq, tokenizer_fr)
    ground_truth_text = translate_sequence(testY[index].flatten(), tokenizer_fr)

    # Return results
    return input_text, predicted_text, ground_truth_text


def calculate_metrics(predicted_text, ground_truth_text):
    results = []
    # WER
    wer = WordErrorRate()
    wer_score = wer(predicted_text, ground_truth_text)
    results.append("WER: " + str(wer_score.item()))
    # BLEU
    bleu = BLEUScore(n_gram=1, smooth=True)
    bleu_score = bleu(predicted_text, ground_truth_text)
    results.append("BLEU: " + str(bleu_score.item()))
    # ROUGE score
    rouge = ROUGEScore()
    rouge_score = rouge(predicted_text, ground_truth_text)
    results.append("ROUGE: " + str(rouge_score['rouge1_fmeasure'].item()))
    # TER
    ter = TranslationEditRate()
    ter_score = ter(predicted_text, ground_truth_text)
    results.append("TER: " + str(ter_score.item()))
    # BERT (not working, as the length of the predicted and the original text are not of the same length)
    bert = BERTScore()
    if not predicted_text or not len(predicted_text) == len(ground_truth_text):
        bert_score = "None"
        results.append("BERT: " + str(bert_score))
    else:
        bert_score = bert(predicted_text, ground_truth_text)
        avg_bert_score_precision = torch.mean(list(bert_score.values())[0])
        avg_bert_score_recall = torch.mean(list(bert_score.values())[1])
        avg_bert_score_f1 = torch.mean(list(bert_score.values())[2])
        bert_results = [avg_bert_score_precision.item(), avg_bert_score_recall.item(), avg_bert_score_f1.item()]
        results.append("BERT: " + str(bert_results))
    return results


def create_metric_file(method):
    path = './results/evaluation/eval_metrics_' + method + '.txt'
    if not os.path.exists(path):
        with open(path, 'w'): pass


def create_summary_file(method):
    path = './results/model_summary/model_summary_' + method + '.txt'
    if not os.path.exists(path):
        with open(path, 'w'): pass

def create_training_file(method):
    path = './results/training_log/training_log_' + method + '.csv'
    if not os.path.exists(path):
        with open(path, 'w'): pass

def create_weight_file_check_train(method):
    path = './results/weights/weights_' + method + '.best.h5'
    if not os.path.exists(path):
        with open(path, 'w'): pass
        return True
    else:
        user_input = input('Would you like to use the existing weights for the model ' + method + '? (y/n): ')
        if user_input.lower() == 'y' or user_input.lower() == 'yes':
            print("Use existing weights")
            return False
        elif user_input.lower() == 'n' or user_input.lower() == 'no':
            print("Train model")
            return True
        else:
            print("No valid answer. Use existing weights")
            return False

def average_metric_results(metric_results):
    avg_results = []
    avg_results.append(str(metric_results[0]) + "\n")
    WER_list = []
    BLEU_list = []
    ROUGE_list = []
    TER_list = []
    BERT_list = []
    for result in metric_results[1:]:
        WER_list.append(float(result[0].split("WER: ")[1]))
        BLEU_list.append(float(result[1].split("BLEU: ")[1]))
        ROUGE_list.append(float(result[2].split("ROUGE: ")[1]))
        TER_list.append(float(result[3].split("TER: ")[1]))
        bert_value = result[4].split("BERT: ")[1]
        if bert_value != 'None':
            BERT_list.append(ast.literal_eval(result[4].split("BERT: ")[1])[0])
    average_WER = sum(WER_list) / len(WER_list)
    avg_results.append("WER: " + str(average_WER))
    average_BLEU = sum(BLEU_list) / len(BLEU_list)
    avg_results.append("BLEU: " + str(average_BLEU))
    average_ROUGE = sum(ROUGE_list) / len(ROUGE_list)
    avg_results.append("ROUGE: " + str(average_ROUGE))
    average_TER = sum(TER_list) / len(TER_list)
    avg_results.append("TER: " + str(average_TER))
    average_BERT = None
    if len(BERT_list) != 0:
        average_BERT = sum(BERT_list) / len(BERT_list)
    avg_results.append("BERT: " + str(average_BERT))
    return avg_results

def write_metric_results(results, method):
    with open(f'./results/evaluation/eval_metrics_{method}.txt', 'w', encoding='utf-8') as file:
        #for result in results:
        file.write(str(results) + "\n")


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
    #else:
        #raise SystemError("GPU device not found")


    # --- 4. We define global variables ---

    EPOCHS = 100
    BATCH_SIZE = 32
    MAX_VOCAB_SIZE_FR = 20500


    # --- 5. We open the data and apply tokenization ---

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


    # --- 6. We load the model ---

    method_name = ['SAN_GRU_Attention',
                   'SAN_GRU_MultiHeadAttention',
                   'SAN_LSTM_Attention',
                   'SAN_LSTM_MultiHeadAttention',
                   'SAN_RNN_Attention',
                   'SAN_RNN_MultiHeadAttention',
                   'SAN_CNNAttention',
                   'SAN_CNN_MultiHeadAttention']
    method_instance = [SAN_GRU_Attention(tokenizer_en, tokenizer_fr, max_len, MAX_VOCAB_SIZE_FR),
                       SAN_GRU_MultiHeadAttention(tokenizer_en, tokenizer_fr, max_len, MAX_VOCAB_SIZE_FR),
                       SAN_LSTM_Attention(tokenizer_en, tokenizer_fr, max_len, MAX_VOCAB_SIZE_FR),
                       SAN_LSTM_MultiHeadAttention(tokenizer_en, tokenizer_fr, max_len, MAX_VOCAB_SIZE_FR),
                       SAN_RNN_Attention(tokenizer_en, tokenizer_fr, max_len, MAX_VOCAB_SIZE_FR),
                       SAN_RNN_MultiHeadAttention(tokenizer_en, tokenizer_fr, max_len, MAX_VOCAB_SIZE_FR),
                       SAN_CNN_Attention(tokenizer_en, tokenizer_fr, max_len, MAX_VOCAB_SIZE_FR),
                       SAN_CNN_MultiHeadAttention(tokenizer_en, tokenizer_fr, max_len, MAX_VOCAB_SIZE_FR)]

    # Shared Callbacks
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', verbose=1)

    for i in range(len(method_name)):
        print(method_name[i])
        current_model = method_instance[i].build_model()
        print(current_model.summary())
        create_summary_file(method_name[i])
        with open(f'./results/model_summary/model_summary_{method_name[i]}.txt', 'w', encoding='utf-8') as file:
            with redirect_stdout(file):
                current_model.summary()


        # --- 7. We train the model ---

        train_model = create_weight_file_check_train(method_name[i])
        if train_model == True:
            create_training_file(method_name[i])
            checkpoint = ModelCheckpoint(f'./results/weights/weights_{method_name[i]}.best.h5', monitor='val_accuracy',
                                         verbose=1, save_best_only=True, mode='max')
            csv_logger = TimedCSVLogger(f'./results/training_log/training_log_{method_name[i]}.csv', append=True)

            if method_name[i] == 'SAN_CNN_Attention' or method_name[i] == 'SAN_CNN_MultiHeadAttention' or method_name[i] == 'SAN_GRU_Attention' or method_name[i] == 'SAN_GRU_MultiHeadAttention' or method_name[i] == 'SAN_LSTM_Attention' or method_name[i] == 'SAN_LSTM_MultiHeadAttention' or method_name[i] == 'SAN_RNN_Attention' or method_name[i] == 'SAN_RNN_MultiHeadAttention':
                current_model.fit([trainX, np.squeeze(trainY, axis=-1)], trainY,
                                  epochs=EPOCHS,
                                  validation_split=0.2,
                                  batch_size=BATCH_SIZE,
                                  callbacks=[checkpoint, csv_logger, early_stopping])
            else:
                current_model.fit(trainX, trainY,
                                  epochs=EPOCHS,
                                  validation_data=(testX, testY),
                                  batch_size=BATCH_SIZE,
                                  callbacks=[checkpoint, csv_logger, early_stopping])
        else:
            current_model = tf.keras.saving.load_model("./results/weights/weights_" + method_name[i] + ".best.h5")

        # --- 8. We test the model (Change for more meaningful metrics like BLEU) ---

        all_predictions = []
        for j in range(20):
            if method_name[i] == 'SAN_CNN_Attention' or method_name[i] == 'SAN_CNN_MultiHeadAttention' or method_name[i] == 'SAN_GRU_Attention' or method_name[i] == 'SAN_GRU_MultiHeadAttention' or method_name[i] == 'SAN_LSTM_Attention' or method_name[i] == 'SAN_LSTM_MultiHeadAttention' or method_name[i] == 'SAN_RNN_Attention' or method_name[i] == 'SAN_RNN_MultiHeadAttention':
                input_text, predicted_text, ground_truth_text = predict_and_compare_auto_en(index=j, testX=testX,
                                                                                            testY=np.squeeze(testY,
                                                                                                             axis=-1),
                                                                                            model=current_model,
                                                                                            tokenizer_en=tokenizer_en,
                                                                                            tokenizer_fr=tokenizer_fr)
            else:
                input_text, predicted_text, ground_truth_text = predict_and_compare(index=j, testX=testX,
                                                                                    model=current_model,
                                                                                    tokenizer_en=tokenizer_en,
                                                                                    tokenizer_fr=tokenizer_fr)
            all_predictions.append((input_text, predicted_text, ground_truth_text))

        # Writing predictions to a text file
        with open(f'./results/predictions/model_predictions_{method_name[i]}.txt', 'w', encoding='utf-8') as file:
            for input_text, predicted_text, ground_truth in all_predictions:
                file.write("Input (English): " + input_text + "\n")
                file.write("Predicted (French): " + predicted_text + "\n")
                file.write("Ground Truth (French): " + ground_truth + "\n")
                file.write("----------\n")

        with open(f'./results/predictions/model_predictions_{method_name[i]}.txt', 'r', encoding='utf-8') as file:
            lines = file.readlines()
            create_metric_file(method_name[i])
            metric_results = [str(method_name[
                                      i]) + ": WER: 1(low), 0(high); BLEU: 0(low), 1(high); TER: 100%(low), 0%(high), BERT: 0(low), 1(high)"]
            for line in lines:
                if "Predicted (French): " in line:
                    prediction = line.split("Predicted (French): ")[1]
                elif "Ground Truth (French): " in line:
                    ground_truth = line.split("Ground Truth (French): ")[1]
                elif "----------" in line:
                    metric_result = calculate_metrics(prediction, ground_truth)
                    metric_results.append(metric_result)
            results = average_metric_results(metric_results)
            write_metric_results(results, method_name[i])
            print("All metrics are calculated for " + method_name[i])
