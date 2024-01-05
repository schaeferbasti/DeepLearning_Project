from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Conv1D, Concatenate, MaxPooling1D, Dense, Dropout, Flatten, Input, \
    RepeatVector, Reshape


class CNN_Auto_Bigger:
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
        # Adding conv and pool layers in the encoder
        #encoder_cnn1 = Conv1D(16384, kernel_size=50, padding='same', activation='relu')(enc_emb)
        #encoder_cnn2 = Conv1D(8192, kernel_size=20, padding='same', activation='relu')(encoder_cnn1)
        #encoder_cnn3 = Conv1D(4096, kernel_size=10, padding='same', activation='relu')(encoder_cnn2)
        #encoder_cnn4 = Conv1D(2048, kernel_size=10, padding='same', activation='relu')(encoder_cnn3)
        encoder_cnn5 = Conv1D(1024, kernel_size=5, padding='same', activation='relu')(enc_emb)
        encoder_cnn6 = Conv1D(512, kernel_size=5, padding='same', activation='relu')(encoder_cnn5)
        encoder_cnn7 = Conv1D(256, kernel_size=5, padding='same', activation='relu')(encoder_cnn6)
        encoder_cnn8 = Conv1D(128, kernel_size=3, padding='same', activation='relu')(encoder_cnn7)
        encoder_cnn9 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(encoder_cnn8)
        encoder_cnn10 = Conv1D(32, kernel_size=3, padding='same', activation='relu')(encoder_cnn9)

        # Decoder
        decoder_inputs = Input(shape=(None,))
        dec_emb_layer = Embedding(input_dim=len(self.tokenizer_fr.word_index) + 1, output_dim=32)
        dec_emb = dec_emb_layer(decoder_inputs)

        # Adding conv and pool layers + encoder in the decoder
        #decoder_cnn1 = Conv1D(16384, kernel_size=50, padding='same', activation='relu')(dec_emb)
        #encoder_output_repeated = RepeatVector(32)(Flatten()(encoder_cnn3))
        #encoder_output_repeated = Reshape((32, 64))(encoder_output_repeated)
        #merged_input = Concatenate(axis=-1)([decoder_cnn1, encoder_output_repeated])
        #decoder_cnn2 = Conv1D(8192, kernel_size=20, padding='same', activation='relu')(decoder_cnn1)
        #decoder_cnn3 = Conv1D(4096, kernel_size=10, padding='same', activation='relu')(decoder_cnn2)
        #decoder_cnn4 = Conv1D(2048, kernel_size=10, padding='same', activation='relu')(decoder_cnn3)
        decoder_cnn5 = Conv1D(1024, kernel_size=5, padding='same', activation='relu')(dec_emb)
        decoder_cnn6 = Conv1D(512, kernel_size=5, padding='same', activation='relu')(decoder_cnn5)
        decoder_cnn7 = Conv1D(256, kernel_size=5, padding='same', activation='relu')(decoder_cnn6)
        decoder_cnn8 = Conv1D(128, kernel_size=3, padding='same', activation='relu')(decoder_cnn7)
        decoder_cnn9 = Conv1D(64, kernel_size=3, padding='same', activation='relu')(decoder_cnn8)
        decoder_cnn10 = Conv1D(32, kernel_size=3, padding='same', activation='relu')(decoder_cnn9)
        decoder_dense = Dense(self.max_vocab_fr_len + 1, activation='softmax')
        decoder_outputs = decoder_dense(decoder_cnn10)
        # Add to a model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
