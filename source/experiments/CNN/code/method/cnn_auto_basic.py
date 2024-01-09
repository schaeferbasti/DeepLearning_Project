from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Conv1D, Dense, Input


class CNN_Auto_Basic:
    def __init__(self, tokenizer_en, tokenizer_fr, max_len, max_vocab_fr_len):
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr
        self.max_len = max_len
        self.max_vocab_fr_len = max_vocab_fr_len

    def build_model(self):
        # Encoder
        encoder_inputs = Input(shape=(None,))
        enc_emb_layer = Embedding(input_dim=len(self.tokenizer_en.word_index) + 1, output_dim=32)
        enc_emb = enc_emb_layer(encoder_inputs)

        encoder_cnn1 = Conv1D(128, kernel_size=5, padding='same', activation='relu')(enc_emb)
        encoder_cnn2 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(encoder_cnn1)
        encoder_cnn3 = Conv1D(32, kernel_size=3, padding='same', activation='relu')(encoder_cnn2)

        # Decoder
        decoder_inputs = Input(shape=(None,))

        decoder_cnn1 = Conv1D(128, kernel_size=5, padding='same', activation='relu')(encoder_cnn3)
        decoder_cnn2 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(decoder_cnn1)
        decoder_cnn3 = Conv1D(32, kernel_size=3, padding='same', activation='relu')(decoder_cnn2)
        decoder_dense = Dense(self.max_vocab_fr_len + 1, activation='relu')
        decoder_outputs = decoder_dense(decoder_cnn3)

        # Model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
