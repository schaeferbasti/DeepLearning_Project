from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, Activation, MaxPooling1D, Conv1DTranspose, Dense


class CNN_Auto_Complex:
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

        enc_conv_1 = Conv1D(128, 5, padding='same')(enc_emb)
        enc_activ_1 = Activation('relu')(enc_conv_1)
        enc_pool_1 = MaxPooling1D(pool_size=3, strides=1, padding='same')(enc_activ_1)
        enc_conv_2 = Conv1D(64, 3, padding='same')(enc_pool_1)
        enc_activ_2 = Activation('relu')(enc_conv_2)
        enc_pool_2 = MaxPooling1D(pool_size=3, strides=1, padding='same')(enc_activ_2)
        enc_conv_3 = Conv1D(32, 3, padding='same')(enc_pool_2)

        # Decoder
        decoder_inputs = Input(shape=(None,))
        dec_emb_layer = Embedding(input_dim=len(self.tokenizer_fr.word_index) + 1, output_dim=32)
        dec_emb = dec_emb_layer(decoder_inputs)
        dec_conv_1 = Conv1D(128, 5, strides=1, padding='same')(dec_emb)
        dec_activ_1 = Activation('relu')(dec_conv_1)
        dec_conv_2 = Conv1D(64, 3, strides=1, padding='same')(dec_activ_1)
        dec_activ_2 = Activation('relu')(dec_conv_2)
        dec_conv_3 = Conv1D(32, 3, strides=1, padding='same')(dec_activ_2)
        dec_activ_3 = Activation('relu')(dec_conv_3)
        decoder_dense = Dense(self.max_vocab_fr_len + 1, activation='softmax')
        decoder_outputs = decoder_dense(dec_activ_3)

        # Model
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
