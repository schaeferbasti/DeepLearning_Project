from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Attention, Concatenate, Dropout

class SAN_LSTM_Attention:
    def __init__(self, tokenizer_en, tokenizer_fr, max_len, max_vocab_fr_len):
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr
        self.max_len = max_len
        self.max_vocab_fr_len = max_vocab_fr_len

    def build_model(self):
        # Encoder
        encoder_inputs = Input(shape=(self.max_len,))
        encoder_embedding = Embedding(input_dim=len(self.tokenizer_en.word_index) + 1, output_dim=64)(encoder_inputs)
        encoder_rnn1 = LSTM(32, return_sequences=True)(encoder_embedding)
        encoder_dropout1 = Dropout(0.5)(encoder_rnn1)
        encoder_rnn2 = LSTM(32, return_sequences=True)(encoder_dropout1)
        encoder_dropout2 = Dropout(0.5)(encoder_rnn2)
        encoder_rnn3 = LSTM(32, return_sequences=True)(encoder_dropout2)

        # Self-Attention Layer
        encoder_attention = Attention(use_scale=True)([encoder_rnn3, encoder_rnn3])

        # Decoder
        decoder_inputs = Input(shape=(None,))
        decoder_embedding = Embedding(input_dim=len(self.tokenizer_fr.word_index) + 1, output_dim=64)(decoder_inputs)
        decoder_rnn1 = LSTM(32, return_sequences=True)(decoder_embedding)
        decoder_dropout1 = Dropout(0.5)(decoder_rnn1)
        decoder_rnn2 = LSTM(32, return_sequences=True)(decoder_dropout1)
        decoder_dropout2 = Dropout(0.5)(decoder_rnn2)
        decoder_rnn3 = LSTM(32, return_sequences=True)(decoder_dropout2)

        # Self-Attention Layer for Decoder
        decoder_attention = Attention(use_scale=True)([decoder_rnn3, decoder_rnn3])

        # Concatenate Encoder and Decoder Outputs
        merged = Concatenate()([encoder_attention, decoder_attention])

        # Dense Layers
        dense1 = Dense(100, activation='relu')(merged)
        output = Dense(len(self.tokenizer_fr.word_index) + 1, activation='softmax')(dense1)

        # Model
        model = Model([encoder_inputs, decoder_inputs], output)

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
