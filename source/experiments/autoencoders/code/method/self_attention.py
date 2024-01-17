from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Concatenate, Add, LayerNormalization, Dense, Dropout, Flatten, Input, \
    RepeatVector, Reshape, MultiHeadAttention
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# inputs
# input embedding
# positional encoding

# positional encoding
# output embedding
# outputs

class TransformerModel:
    def __init__(self, tokenizer_en, tokenizer_fr, max_vocab_fr_len):
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr
        self.max_len = max_len
        self.max_vocab_fr_len = max_vocab_fr_len
        self.vocab_size = len(tokenizer_en.word_index) + 1
        self.last_attn_scores = None
        self.d_model = 32
        self.dropout_rate = 0.1
        self.add = Add()
        self.layer_normalization = LayerNormalization()

    def positional_encoding(self, length, depth):
        """
        gets the positional encoding of a sentence
        """
        depth = depth/2

        positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
        depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

        angle_rates = 1 / (10000**depths)         # (1, depth)
        angle_rads = positions * angle_rates      # (pos, depth)

        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1)

        return tf.cast(pos_encoding, dtype=tf.float32)

    def pos_embedding(self, x, d_model=32):
        """d_model = output dimension
            x= inputs
        """
        x_embedding = Embedding(input_dim=self.vocab_size, output_dim=d_model, mask_zero=True)(x)
        pos_encoding = self.positional_encoding(length=2048, depth=d_model)
        # compute_mask = embedding.compute_mask(*args, **kwargs)

        length = tf.shape(x)[1]
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x_embedding *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x_embedding + pos_encoding[tf.newaxis, :length, :]
        return x
        

    def build_model(self):
        encoder_inputs = Input(shape=(None,))
        decoder_inputs = Input(shape=(None,))

        # input_pos_emb = self.pos_embedding(encoder_inputs)

        # input_pos_emb = self.pos_embedding(encoder_inputs)
        # output_pos_emb = self.pos_embedding(decoder_inputs)
                
        output_sequence_length = 2000
        output_length = 32

        # inputs posi
        position_embedding_layer = Embedding(output_sequence_length, output_length)
        position_indices = tf.range(output_sequence_length)
        embedded_indices = position_embedding_layer(position_indices)

        position_indices = tf.range(tf.shape(encoder_inputs)[-1])
        embedded_words = Embedding(input_dim=2000, output_dim=32)(encoder_inputs)
        embedded_indices = Embedding(input_dim=32, output_dim=32)(position_indices)
        input_pos_emb = embedded_words + embedded_indices

        # outputs posi
        position_embedding_layer2 = Embedding(output_sequence_length, output_length)
        position_indices2 = tf.range(output_sequence_length)
        embedded_indices2 = position_embedding_layer2(position_indices2)

        position_indices2 = tf.range(tf.shape(encoder_inputs)[-1])
        embedded_words2 = Embedding(input_dim=2000, output_dim=32)(decoder_inputs)
        embedded_indices2 = Embedding(input_dim=32, output_dim=32)(position_indices)
        output_pos_emb = embedded_words2 + embedded_indices2

        global_attention = MultiHeadAttention(num_heads=4,key_dim=32, dropout=0.1)(query=input_pos_emb, value=input_pos_emb, key=input_pos_emb)
        ffn1_1 = Dense(32, activation='relu')(global_attention)
        ffn1_2 = Dense(32)(ffn1_1)
        dropout1 = Dropout(0.1)(ffn1_2)
        add1 = Add()([input_pos_emb, dropout1])
        layer_norm1 = LayerNormalization()(add1)
        # Encoder end

        # Decoder
        # casual attention
        casual_attn = MultiHeadAttention(num_heads=4,key_dim=32, dropout=0.1)(query=output_pos_emb, value=output_pos_emb,
                                                key=output_pos_emb, use_causal_mask=True)
        add2 = Add()([output_pos_emb, casual_attn])
        layer_norm2 = LayerNormalization()(add2)

        # cross_attention
        cross_attn = MultiHeadAttention(num_heads=4,key_dim=32, dropout=0.1)(query=layer_norm2, value=layer_norm1, key=layer_norm1,
                                                                    #   return_attention_scores=True
                                                )
        add3 = Add()([layer_norm2, cross_attn])
        layer_norm3 = LayerNormalization()(add3)

        ffn2_1 = Dense(32, activation='relu')(layer_norm3)
        ffn2_2 = Dense(32)(ffn2_1)
        dropout2 = Dropout(0.1)(ffn2_2)
        add4 = Add()([layer_norm3, dropout2])
        layer_norm4 = LayerNormalization()(add4)
        # Decoder End

        # Build Final Model
        model = Model([input_pos_emb, output_pos_emb], layer_norm4)

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    


        return model




df = pd.read_csv('preprocessed_data.csv')
df = df[:121]

tokenizer_en = Tokenizer()
tokenizer_en.fit_on_texts(df['en_tokens'])
tokenizer_fr = Tokenizer(num_words=20500 + 1)
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


helper = TransformerModel(tokenizer_en, tokenizer_fr, 20500)
helper.build_model()