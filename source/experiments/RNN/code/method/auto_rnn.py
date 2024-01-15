from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, Dense, Dropout

class EncoderDecoderRNNModel:
    def __init__(self, tokenizer_en, tokenizer_fr, max_vocab_fr_len):
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr
        self.max_vocab_fr_len = max_vocab_fr_len

    def build_model(self):
        # Encoder
        encoder_inputs = Input(shape=(None,))
        enc_emb = Embedding(input_dim=len(self.tokenizer_en.word_index) + 1, output_dim=32)(encoder_inputs)
        
        # Adding three SimpleRNN layers in the encoder
        encoder_rnn1 = SimpleRNN(32, return_sequences=True)(enc_emb)
        encoder_dropout1 = Dropout(0.5)(encoder_rnn1)
        encoder_rnn2 = SimpleRNN(32, return_sequences=True)(encoder_dropout1)
        encoder_dropout2 = Dropout(0.5)(encoder_rnn2)
        _, state = SimpleRNN(32, return_state=True)(encoder_dropout2)

        # Decoder
        decoder_inputs = Input(shape=(None,))
        dec_emb_layer = Embedding(input_dim=len(self.tokenizer_fr.word_index) + 1, output_dim=32)
        dec_emb = dec_emb_layer(decoder_inputs)
        
        # Adding three SimpleRNN layers in the decoder
        decoder_rnn1 = SimpleRNN(32, return_sequences=True, return_state=True)(dec_emb, initial_state=state)
        decoder_dropout1 = Dropout(0.5)(decoder_rnn1[0])  # Dropout after the first RNN layer
        decoder_rnn2 = SimpleRNN(32, return_sequences=True)(decoder_dropout1)
        decoder_dropout2 = Dropout(0.5)(decoder_rnn2)  # Another dropout layer
        decoder_rnn3 = SimpleRNN(16, return_sequences=True)(decoder_dropout2)
        
        decoder_dense = Dense(self.max_vocab_fr_len + 1, activation='softmax')
        decoder_outputs = decoder_dense(decoder_rnn3)

        # Build Final Model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
