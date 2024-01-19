"""
took a pretrained bart model, randomly
initialized the weights and trained it on  our data
"""

import tensorflow as tf
import numpy as np
from transformers import TFBartForConditionalGeneration, BartConfig


class BartBasic:
    def __init__(self, tokenizer_en, tokenizer_fr, max_len, max_vocab_fr_len):
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr
        self.max_len = max_len
        self.max_vocab_fr_len = max_vocab_fr_len

    def build_model(self):

        # Define a random configuration
        random_config = BartConfig.from_dict({
            "vocab_size": self.max_vocab_fr_len,
            "d_model": 512,  # Set other hyperparameters as needed
            "num_layers": 6,
            "num_heads": 8,
            "max_length": 1024,
        })

        # Model
        # Build the TFBartForConditionalGeneration model with random initialization
        model = TFBartForConditionalGeneration.from_config(random_config)

        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])

        return model

