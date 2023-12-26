from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout

class SimpleRNNModel:
    def __init__(self, tokenizer_en, tokenizer_fr, max_len):
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr
        self.max_len = max_len

    def build_model(self):
        # Define structure of the model
        model = Sequential()
        model.add(Embedding(input_dim=len(self.tokenizer_en.word_index) + 1, output_dim=64, input_length=self.max_len))
        model.add(SimpleRNN(64, return_sequences=True))
        model.add(Dropout(0.5)) 
        model.add(SimpleRNN(64, return_sequences=True)) 
        model.add(Dropout(0.5)) 
        model.add(SimpleRNN(32, return_sequences=True)) 
        model.add(Dense(32, activation='relu'))
        model.add(Dense(len(self.tokenizer_fr.word_index) + 1, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model