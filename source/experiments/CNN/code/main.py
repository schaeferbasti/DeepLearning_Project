# --- 1. We import the libraries we need ---
import numpy as np
import tensorflow as tf
import argparse
import pandas as pd
import time
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, Callback
from method.cnn_bytenet import CNN_ByteNet
from source.experiments.CNN.code.method.cnn_auto_basic import CNN_Auto_Basic
from source.experiments.CNN.code.method.cnn_basic import CNN_Basic


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
    # else:
    # raise SystemError("GPU device not found")

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
    method_name = ['CNN_Auto_Basic', 'CNN_Basic']  # , 'CNN_ByteNet']
    method_instance = [CNN_Auto_Basic(tokenizer_en, tokenizer_fr, max_len, MAX_VOCAB_SIZE_FR),
                       CNN_Basic(tokenizer_en, tokenizer_fr, max_len,
                                 MAX_VOCAB_SIZE_FR)]  # , CNN_ByteNet(MAX_VOCAB_SIZE_FR)]

    # Shared Callbacks
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max', verbose=1)

    for i in range(len(method_name)):
        current_model = method_instance[i].build_model()

        # --- 5. We train the model ---
        checkpoint = ModelCheckpoint(f'./results/weights/weights_{method_name[i]}.best.h5', monitor='val_accuracy',
                                     verbose=1, save_best_only=True, mode='max')
        csv_logger = TimedCSVLogger(f'./results/training_log/training_log_{method_name[i]}.csv', append=True)

        if method_name[i] == 'CNN_ByteNet':
            run_CNN_ByteNet()
        elif method_name[i] == 'CNN_Auto_Basic':
            current_model.fit([trainX, np.squeeze(trainY, axis=-1)], np.expand_dims(trainY, -1), epochs=EPOCHS,
                              validation_split=0.2,
                              batch_size=BATCH_SIZE, callbacks=[checkpoint, csv_logger, early_stopping])
        else:
            current_model.fit(trainX, trainY, epochs=EPOCHS, validation_data=(testX, testY), batch_size=BATCH_SIZE,
                              callbacks=[checkpoint, csv_logger, early_stopping])

        # --- 6. We test the model (Change for more meaningful metrics like BLEU) ---

        all_predictions = []
        for j in range(5):
            if method_name[i] == 'CNN_Auto_Basic':
                input_text, predicted_text, ground_truth_text = predict_and_compare_auto_en(index=j, testX=testX,
                                                                                            testY=testY,
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


def sample_top(a=[], top_k=10):
    idx = np.argsort(a)[::-1]
    idx = idx[:top_k]
    probs = a[idx]
    probs = probs / np.sum(probs)
    choice = np.random.choice(idx, p=probs)
    return choice


def run_CNN_ByteNet():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning Rate')
    parser.add_argument('--bucket_quant', type=int, default=50,
                        help='Bucket Quant')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Momentum for Adam Update')
    parser.add_argument('--resume_model', type=str, default=None,
                        help='Pre-Trained Model Path, to resume from')
    parser.add_argument('--sample_every', type=int, default=500,
                        help='Sample generator output every x steps')
    parser.add_argument('--summary_every', type=int, default=50,
                        help='Sample generator output every x steps')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions')
    args = parser.parse_args()

    translator_model = CNN_ByteNet(MAX_VOCAB_SIZE_FR)
    translator_model.build_options()

    optim = tf.keras.optimizers.Adam(args.learning_rate)

    translator_model.build_translator(reuse=True)
    translator_model.build_model(MAX_VOCAB_SIZE_FR)
    merged_summary = tf.compat.v1.summary.merge_all()

    sess = tf.compat.v1.InteractiveSession()
    tf.compat.v1.initialize_all_variables().run()
    saver = tf.compat.v1.train.Saver()

    if args.resume_model:
        saver.restore(sess, args.resume_model)

    step = 0
    for epoch in range(EPOCHS):
        batch_no = 0
        start = time.process_time()

        _, loss, prediction = sess.run(
            [optim, translator_model.loss, translator_model.arg_max_prediction],

            feed_dict={
                translator_model.source_sentence: trainX,
                translator_model.target_sentence: trainY,
            })
        end = time.process_time()

        print
        "LOSS: {}\tEPOCH: {}\tBATCH_NO: {}\t STEP:{}\t total_batches:{}\t bucket_size:{}".format(
            loss, epoch, batch_no, step)
        print
        "TIME FOR BATCH", end - start

        batch_no += 1
        step += 1
        if step % args.summary_every == 0:
            [summary] = sess.run([merged_summary], feed_dict={
                translator_model.source_sentence: trainX,
                translator_model.target_sentence: trainY,
            })
            print
            "******"
            print
            "Source ", trainX
            print
            "---------"
            print
            "Target ", trainY
            print
            "----------"
            print
            "Prediction ", prediction
            print
            "******"

        if step % args.sample_every == 0:
            log_file = open('translator_sample.txt', 'wb')
            generated_target = trainY[:, 0:1]
            for col in range(batch_no):
                [probs] = sess.run([translator_model.t_probs],
                                   feed_dict={
                                       translator_model.t_source_sentence: trainX,
                                       translator_model.t_target_sentence: generated_target,
                                   })

                curr_preds = []
                for bi in range(probs.shape[0]):
                    pred_word = sample_top(probs[bi][-1], top_k=args.top_k)
                    curr_preds.append(pred_word)

                generated_target = np.insert(generated_target, generated_target.shape[1],
                                             curr_preds, axis=1)
