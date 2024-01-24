"""
In this file, we want to read the predictions from the predictions file of the model
and use the metrics to evaluate the quality.
"""
import os

from torchmetrics.text import TranslationEditRate
from torchmetrics.text.bert import BERTScore
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.text.wer import WordErrorRate

# Define the Name of the models
method_name = ['CNN_Basic', 'CNN_Auto_Bigger', 'CNN_Auto_Basic']

def calculate_metrics(predicted_text, ground_truth_text):
    results = []
    # WER
    wer = WordErrorRate()
    wer_score = wer(predicted_text, ground_truth_text)
    results.append("WER: " + str(wer_score.item()))
    # BLEU
    bleu = BLEUScore(n_gram=1, smooth=True)
    bleu_score = bleu(predicted_text, ground_truth_text)
    results.append("BLEU: " + str(bleu_score.item()))
    # TER
    ter = TranslationEditRate()
    ter_score = ter(predicted_text, ground_truth_text)
    results.append("TER: " + str(ter_score.item()) + "%")
    # BERT (not working, as the length of the predicted and the original text are not of the same length)
    bert = BERTScore()
    if not predicted_text or not len(predicted_text) == len(ground_truth_text):
        bert_score = "None"
    else:
        bert_score = bert(predicted_text, ground_truth_text)
    results.append("BERT: " + str(bert_score))
    return results

def create_metric_file(method):
    path = './evaluation/eval_metrics_' + method + '.txt'
    if not os.path.exists(path):
        with open(path, 'w'): pass

def write_metric_results(results, method):
    with open(f'./evaluation/eval_metrics_{method}.txt', 'w', encoding='utf-8') as file:
        for result in results:
            file.write(str(result) + "\n")


for i in range(len(method_name)):
    with open(f'./predictions/model_predictions_{method_name[i]}.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
        create_metric_file(method_name[i])
        metric_results = [str(method_name[i]) + ": WER: 1(low), 0(high); BLEU: 0(low), 1(high); TER: 100%(low), 0%(high), BERT: 0(low), 1(high)"]
        for line in lines:
            if "Predicted (French): " in line:
                prediction = line.split("Predicted (French): ")[1]
            elif "Ground Truth (French): " in line:
                ground_truth = line.split("Ground Truth (French): ")[1]
            elif "----------" in line:
                metric_result = calculate_metrics(prediction, ground_truth)
                metric_results.append(str(metric_result))
        write_metric_results(metric_results, method_name[i])
        print("All metrics are calculated for " + method_name[i])



