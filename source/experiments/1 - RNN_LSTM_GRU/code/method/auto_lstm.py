from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout

class EncoderDecoderLSTMModel:
    def __init__(self, tokenizer_en, tokenizer_fr, max_vocab_fr_len):
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr
        self.max_vocab_fr_len = max_vocab_fr_len

    def build_model(self):
        # Encoder
        encoder_inputs = Input(shape=(None,))
        enc_emb = Embedding(input_dim=len(self.tokenizer_en.word_index) + 1, output_dim=32)(encoder_inputs)

        # Adding four LSTM layers in the encoder
        encoder_lstm1 = LSTM(32, return_sequences=True)(enc_emb)
        encoder_dropout1 = Dropout(0.5)(encoder_lstm1)
        encoder_lstm2 = LSTM(32, return_sequences=True)(encoder_lstm1)
        encoder_dropout2 = Dropout(0.5)(encoder_lstm2)
        _, state_h, state_c = LSTM(32, return_state=True)(encoder_dropout2)
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = Input(shape=(None,))
        dec_emb_layer = Embedding(input_dim=len(self.tokenizer_fr.word_index) + 1, output_dim=32)
        dec_emb = dec_emb_layer(decoder_inputs)

        # Adding four LSTM layers in the decoder
        decoder_lstm1 = LSTM(32, return_sequences=True, return_state=True)(dec_emb, initial_state=encoder_states)
        decoder_dropout1 = Dropout(0.5)(decoder_lstm1[0])  # Dropout after the first LSTM layer
        decoder_lstm2 = LSTM(32, return_sequences=True)(decoder_dropout1)
        decoder_dropout2 = Dropout(0.5)(decoder_lstm2)  # Another dropout layer
        decoder_lstm3 = LSTM(16, return_sequences=True)(decoder_dropout2)

        decoder_dense = Dense(self.max_vocab_fr_len + 1, activation='softmax')
        decoder_outputs = decoder_dense(decoder_lstm3)

        # Build Final Model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
