from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, Dense

class Transformer:
    def __init__(self, tokenizer_en, tokenizer_fr, max_len, max_vocab_fr_len):
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr
        self.max_len = max_len
        self.max_vocab_fr_len = max_vocab_fr_len

    def build_model(self):
        # Encoder
        encoder_inputs = Input(shape=(self.max_len,))
        encoder_embedding = Embedding(input_dim=len(self.tokenizer_en.word_index) + 1, output_dim=64)(encoder_inputs)
        # Self-Attention Layer
        encoder_attention = MultiHeadAttention(num_heads=8, key_dim=64)(encoder_embedding, encoder_embedding)

        # Decoder
        decoder_inputs = Input(shape=(None,))
        decoder_embedding = Embedding(input_dim=len(self.tokenizer_fr.word_index) + 1, output_dim=64)(decoder_inputs)
        # Self-Attention Layer for Decoder
        decoder_attention = MultiHeadAttention(num_heads=8, key_dim=64)(decoder_embedding, decoder_embedding)

        # Concatenate Encoder and Decoder Outputs
        merged = MultiHeadAttention(num_heads=8, key_dim=64)(encoder_attention, decoder_attention)
        # Dense Layers
        # dense1 = Dense(100, activation='relu')(merged)
        feed_forward = Dense(len(self.tokenizer_fr.word_index) + 1, activation='softmax')(merged)

        # Model
        model = Model([encoder_inputs, decoder_inputs], feed_forward)

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
