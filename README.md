# DeepLearning_Project

---- Exercise: ----

The objective is to extensively study the available deep learning solutions for the text to text translation problem using the french to English text dataset from the following kaggle competition:

from datasets import load_dataset
translation_dataset = load_dataset(’Nicolas-BZRD/Parallel_Global_Voices_English_French’, split=’train’).to_pandas()
translation_dataset.head()

---- Tasks: ----

1. Exploratory data analysis of the dataset -> Look at the amount of words, most frequent ones, distribution if possible. -> A notebook explaining all insights found. (Bastian)
2. Explore pre-processing of the data, word embedding, and data augmentation -> Bag of words, Word2Vec, TF-IDF, etc. -> A notebook explaining all the procedures done. (Gloria)
3. Explore architectures and experiments to solve the problem (For each procedure detail architecture, metrics, tuning, results, challenges faced and extra-elements implemented for example: attention-based transformation). (Felipe)
-> For each experiment try to create a notebook explaining everything.

Some ideas for experiments (Choose one and write in the Discord group which one you picked so we don't do the same thing):
- data processing.
- data augmentation.
- supervised pre-training with auto-encoders.
- convolutional models (yes, there exists convolutions for text models).
- attention-based (transformer) models (As the one we saw in the TP's with BERT).
- classification methodologies.
- generation techniques.