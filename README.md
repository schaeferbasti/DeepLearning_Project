# Deep Learning II Project - 2024

## Overview

Welcome to the Deep Learning II Project repository. This project is part of the Master's program in Machine Learning and Data Mining at Jean Monnet University. Our team is focused on extensively study the available deep learning solutions for the text to text translation problem using the French to English text dataset from a kaggle competition.

## Dataset

The dataset can be obtained from the following link: https://huggingface.co/datasets/Nicolas-BZRD/Parallel_Global_Voices_English_French. Also, using the following code:

```
from datasets import load_dataset

translation_dataset = load_dataset("Nicolas-BZRD/Parallel_Global_Voices_English_French", split="train").to_pandas()
translation_dataset.head()
```

## Project Description

The primary challenge addressed in this project is exploring different architectures in order to carry out machine translation from English to French. For this reason, the project can be decomposed in three main steps:

1. **Data Pre-processing:** Find the best way to pre-process the data in order to feed them to the network architectures. 

2. **Model Definition:** Design and implement different model architectures using RNNs, CNNs and SANs principles.

3. **Model Comparison:** Compare the models with approved machine translation metrics such as: WER, BLEU, ROUGE, TER and BERT-Score.

## Repository Contents

- `source/`: Contains all source code (both code and notebooks) for the project.
  - Analysis made throughout the project such as DEA and correlations.
  - Experiments done with different models architectures.
  - Preprocessing utilities.

### Prerequisites

- See `requirements.txt` for a list of required Python packages.

## Usage

In order to test a simple model in our repository you can follow the following steps. This first example uses a condensed corpus for testing purposes.

1. We install the dependencies:

```
pip install -r requirements.txt
```

2. We move the `source/preprocessing/preprocessed_data.csv` file to our desired folder, in this example we'll use the first experiment:

```
mv ./source/preprocessing/preprocessed_data.csv ./source/experiments/1 - RNN_LSTM_GRU/code/
```

3. We execute the main script.:
```
python ./source/experiments/1 - RNN_LSTM_GRU/code/main.py
```

This second example, generates the whole corpus and executes the first experiment.

1. We install the dependencies:

```
pip install -r requirements.txt
```

2. We generate and move the data file to the desired folder, in this example we'll use the first experiment:

```
python ./source/preprocessing/code/pre_processing.py

mv ./source/preprocessing/code/preprocessed_data.csv ./source/experiments/1 - RNN_LSTM_GRU/code/
```

3. We execute the main script.:
```
python ./source/experiments/1 - RNN_LSTM_GRU/code/main.py
```

## Authors and Acknowledgment

- Made by Felipe Jaramillo Cortes, Bastian Schäfer and Gloria Isedu.
- Supervised by Amaury Habrard and Sri Kalidindi.