# cited from https://medium.com/@dhirensk/tensorflow-addons-seq2seq-example-using-attention-and-beam-search-9f463b58bc6b

import tensorfow as tf
import tensorflow_addons as tfa
import numpy as np


def translate(model, sentence_en):
    translation = ""
    for word_idx in range(model.max_sentence_len):
        X_encoder = np.array([sentence_en])
        X_decoder = np.array(["startofseq " + translation])
        # Last token's probas.
        y_proba = model.predict((X_encoder, X_decoder), verbose=0)[0, word_idx]
        predicted_word_id = np.argmax(y_proba)
        predicted_word = model.vectorization_fr.get_vocabulary()[predicted_word_id]
        if predicted_word == "endofseq":
            break
        translation += " " + predicted_word
    return translation.strip()


# Here, mask is a zero-one matrix of the same size as decoder_outputs.
#  It masks padding positions outside of the target sequence lengths with values 0.
def loss_function(y_pred, y):
   
    #shape of y [batch_size, ty]
    #shape of y_pred [batch_size, Ty, output_vocab_size] 
    sparse_cat_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                                  reduction='none')
    loss = sparse_cat_cross_entropy(y_pred=y_pred, y_true=y)
    #skip loss calculation for padding sequences i.e. y = 0 
    #[ ,How, are, you, today, 0, 0, 0, 0 ....]
    #[ 1, 234, 3, 423, 3344, 0, 0 ,0 ,0, 2 ]
    # y is a tensor of [batch_size,Ty] . Create a mask when [y=0]
    # mask the loss when padding sequence appears in the output sequence
    mask = tf.logical_not(tf.math.equal(y,0))   #output 0 for y=0 else output 1
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = mask* loss
    loss = tf.reduce_mean(loss)
    return loss



# ENCODER
class EncoderNN(tf.keras.Model):
    def __init__(self, input_dim, output_dim, dim_rnn_output):
        super().__init__()
        self.encoder_embedding = tf.keras.layers.Embedding(input_dim=input_dim,
                                                           output_dim=output_dim)
        self.encoder_rnnlayer = tf.keras.layers.LSTM(dim_rnn_output,return_sequences=True, 
                                                     return_state=True )


# DECODER
class DecoderNN(tf.keras.Model):
    def __init__(self, input_dim, output_dim, dim_rnn_output, attention_mech_depth, batch_size, max_en_X):
        super().__init__()
        self.decoder_embedding = tf.keras.layers.Embedding(input_dim=input_dim,
                                                           output_dim=output_dim) 
        self.dense_layer = tf.keras.layers.Dense(input_dim)
        self.decoder_rnncell = tf.keras.layers.LSTMCell(dim_rnn_output)
        
        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(attention_mech_depth,
                                                                  None,memory_sequence_length=batch_size*[max_en_X])
        self.rnn_cell =  self.build_rnn_cell(attention_layer_size=attention_mech_depth)
        self.decoder = tfa.seq2seq.BasicDecoder(self.rnn_cell, sampler= self.sampler,
                                                output_layer=self.dense_layer)
        
    
    def build_attention_mechanism(self, units,memory, memory_sequence_length):
        return tfa.seq2seq.LuongAttention(units, memory=memory, 
                                          memory_sequence_length=memory_sequence_length)

    # wrap decoder rnn cell  
    def build_rnn_cell(self, attention_layer_size):
        return tfa.seq2seq.AttentionWrapper(self.decoder_rnncell, self.attention_mechanism,
                                                attention_layer_size=attention_layer_size)
        
    def build_decoder_initial_state(self, batch_size, encoder_state,Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(batch_size = batch_size, 
                                                                dtype = Dtype)
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state) 
        return decoder_initial_state

    
class BidirectionalEncoderDecoderWithAttention(tf.keras.Model):
    def __init__(
        self,
        vocabulary_size=5000,
        max_sentence_len=50,
        embedding_size=256,
        n_units_lstm=512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_sentence_len = max_sentence_len

        self.vectorization_en = layers.TextVectorization(
            vocabulary_size, output_sequence_length=max_sentence_len
        )
        self.vectorization_fr = layers.TextVectorization(
            vocabulary_size, output_sequence_length=max_sentence_len
        )

        self.encoder_embedding = layers.Embedding(
            vocabulary_size, embedding_size, mask_zero=True
        )
        self.decoder_embedding = layers.Embedding(
            vocabulary_size, embedding_size, mask_zero=True
        )

        self.encoder = layers.Bidirectional(
            layers.LSTM(n_units_lstm // 2, return_sequences=True, return_state=True)
        )
        self.decoder = layers.LSTM(n_units_lstm, return_sequences=True)
        self.attention = layers.Attention()
        self.output_layer = layers.Dense(vocabulary_size, activation="softmax")

    def call(self, inputs):
        encoder_inputs, decoder_inputs = inputs

        encoder_input_ids = self.vectorization_en(encoder_inputs)
        decoder_input_ids = self.vectorization_fr(decoder_inputs)

        encoder_embeddings = self.encoder_embedding(encoder_input_ids)
        decoder_embeddings = self.decoder_embedding(decoder_input_ids)

        # The final hidden state of the encoder, representing the entire
        # input sequence, is used to initialize the decoder.
        encoder_output, *encoder_state = self.encoder(encoder_embeddings)
        encoder_state = [
            tf.concat(encoder_state[0::2], axis=-1),  # Short-term state (0 & 2).
            tf.concat(encoder_state[1::2], axis=-1),  # Long-term state (1 & 3).
        ]
        decoder_output = self.decoder(decoder_embeddings, initial_state=encoder_state)
        attention_output = self.attention([decoder_output, encoder_output])

        return self.output_layer(attention_output)