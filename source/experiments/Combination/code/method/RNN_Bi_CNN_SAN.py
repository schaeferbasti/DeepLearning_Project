from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, SimpleRNN, Bidirectional, Attention, Concatenate, Dense

class SAN_CNN_Attention:
    def __init__(self, tokenizer_en, tokenizer_fr, max_len, max_vocab_fr_len):
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr
        self.max_len = max_len
        self.max_vocab_fr_len = max_vocab_fr_len

    def build_model(self):
        # Encoder
        encoder_inputs = Input(shape=(self.max_len,))
        encoder_embedding = Embedding(input_dim=len(self.tokenizer_en.word_index) + 1, output_dim=64)(encoder_inputs)
        encoder_rnn1 = Bidirectional(SimpleRNN(64, return_sequences=True))(encoder_embedding)
        encoder_rnn2 = Bidirectional(SimpleRNN(64, return_sequences=True))(encoder_rnn1)
        encoder_rnn3 = Bidirectional(SimpleRNN(64, return_sequences=True))(encoder_rnn2)
        encoder_conv1 = Conv1D(256, kernel_size=8, padding='same', activation='relu')(encoder_rnn3)
        encoder_conv2 = Conv1D(128, kernel_size=5, padding='same', activation='relu')(encoder_conv1)
        encoder_conv3 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(encoder_conv2)

        # Self-Attention Layer
        encoder_attention = Attention(use_scale=True)([encoder_conv3, encoder_conv3])

        # Decoder
        decoder_inputs = Input(shape=(None,))
        decoder_embedding = Embedding(input_dim=len(self.tokenizer_fr.word_index) + 1, output_dim=64)(decoder_inputs)
        decoder_rnn1 = Bidirectional(SimpleRNN(64, return_sequences=True))(decoder_embedding)
        decoder_rnn2 = Bidirectional(SimpleRNN(64, return_sequences=True))(decoder_rnn1)
        decoder_rnn3 = Bidirectional(SimpleRNN(64, return_sequences=True))(decoder_rnn2)
        decoder_conv1 = Conv1D(256, kernel_size=8, padding='same', activation='relu')(decoder_rnn3)
        decoder_conv2 = Conv1D(128, kernel_size=5, padding='same', activation='relu')(decoder_conv1)
        decoder_conv3 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(decoder_conv2)

        # Self-Attention Layer for Decoder
        decoder_attention = Attention(use_scale=True)([decoder_conv3, decoder_conv3])

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