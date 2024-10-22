from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, Dense


class CNN_Basic:
    def __init__(self, tokenizer_en, tokenizer_fr, max_len, max_vocab_fr_len):
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr
        self.max_len = max_len
        self.max_vocab_fr_len = max_vocab_fr_len

    def build_model(self):
        # Model
        model = Sequential()
        model.add(Embedding(input_dim=len(self.tokenizer_en.word_index) + 1, output_dim=64, input_length=self.max_len))
        model.add(Conv1D(256, kernel_size=8, padding='same', activation='relu'))
        model.add(Conv1D(128, kernel_size=5, padding='same', activation='relu'))
        model.add(Conv1D(64, kernel_size=3, padding='same', activation='relu'))
        model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
        model.add(Dense(100, activation='softmax'))
        model.add(Dense(len(self.tokenizer_fr.word_index) + 1)),

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
