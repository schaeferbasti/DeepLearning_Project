from __future__ import absolute_import, division, print_function
%matplotlib inline

from keras.datasets import mnist
# from keras.layers import Input, Dense, Reshape, Flatten, Dropout, MaxPooling2D, Lambda
# from keras.layers import BatchNormalization, Activation, ZeroPadding2D
# from keras.layers.advanced_activations import LeakyReLU
# from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
# import keras.losses as Losses
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
import tensorflow as tf
#import keras.backend as K

import random
import matplotlib.pyplot as plt
import sys
import numpy as np


def positional_encoding(length, depth):
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


class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    """
    computes embedding mask
    """
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    """
    execute positional embedding
    """
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x


class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
  def call(self, x, context):
    """
    execute cross attention
    """
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        return_attention_scores=True)

    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x
  

class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    """
    execute global self attention
    """
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
  

class FeedForward(tf.keras.layers.Layer):

  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    """
    execute feed forward network
    """
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x


class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    """
    execure encoder layer with self attention
    """
    x = self.self_attention(x)
    x = self.ffn(x)
    return x


class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, vocab_size, dropout_rate=0.1):
    """
    Args: 
          num_layers - desired number of encoder layers
          dff - dimensionality of outter space
          vocab_size - input dimensin / vocabulary size
          dropout_rate - dropout probability.
    """
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(
        vocab_size=vocab_size, d_model=d_model)

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    """
    execute encoder 
    """

    # do a positional embedding
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

    # Add dropout.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x  # Shape `(batch_size, seq_len, d_model)`.
  

# Instantiate the encoder.
sample_encoder = Encoder(num_layers=4,
                         d_model=512,
                         num_heads=8,
                         dff=2048,
                         vocab_size=8500)


# # Create training and validation set batches.
# train_batches = make_batches(train_examples)
# val_batches = make_batches(val_examples)

# for (en, fr), fr_labels in train_batches.take(1):
#   break

# print(en.shape)
# print(fr.shape)
# print(fr_labels.shape)


class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
#         encoder.attention_dropout_rate: 0.3
#   encoder.attention_type: dot_product
#   encoder.ffn_activation: relu
#   encoder.ffn_dropout_rate: 0.3
#   encoder.filter_size: 4096
#   encoder.hidden_size: 1024
#   encoder.layer_postprocess_dropout_rate: 0.3
#   encoder.layer_postprocess_epsilon: 1.0e-06
#   encoder.num_attention_heads: 16
#   encoder.num_layers: 12
#   encoder.post_normalize: false
        self.encoder = Sequential([
#             layers.Dense(128, activation='relu')(input_img)
# layers.Dense(64, activation='relu')(encoded)
        layers.Flatten(),
        layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
        layers.Dense(tf.math.reduce_prod(shape), activation='sigmoid'),
        layers.Reshape(shape)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

shape = x_test.shape[1:]
latent_dim = 64
autoencoder = Autoencoder(latent_dim, shape)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.fit(x_train, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test, x_test))


input_img = Input(shape=(28,28,1,))
flat = Flatten()(input_img)
encoded = Dense(128, activation='relu')(flat)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)
decoded = Reshape(target_shape = (28,28,1,))(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

# train autoencoder
no_of_epochs = 10
n = 10  # how many digits we will display
# plt.figure(figsize=(20, 2))
# print('original images:')
# print_imgs(x_test[:10])
    
