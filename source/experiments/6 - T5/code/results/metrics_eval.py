"""
In this file, we want to read the predictions from the predictions file of the model
and use the metrics to evaluate the quality.
"""
import os
import ast
import torch
from torchmetrics.text.wer import WordErrorRate
from torchmetrics.text.bleu import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text import TranslationEditRate
from torchmetrics.text.bert import BERTScore

# Define the Name of the models
method_name = ['t5_base_transformer_mt', 't5_transformer_mt']

# Define the path to the predictions file
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
    # ROUGE score
    rouge = ROUGEScore()
    rouge_score = rouge(predicted_text, ground_truth_text)
    results.append("ROUGE: " + str(rouge_score['rouge1_fmeasure'].item()))
    # TER
    ter = TranslationEditRate()
    ter_score = ter(predicted_text, ground_truth_text)
    results.append("TER: " + str(ter_score.item()))
    # BERT (not working, as the length of the predicted and the original text are not of the same length)
    bert = BERTScore()
    if not predicted_text or not len(predicted_text) == len(ground_truth_text):
        bert_score = "None"
        results.append("BERT: " + str(bert_score))
    else:
        bert_score = bert(predicted_text, ground_truth_text)
        avg_bert_score_precision = torch.mean(list(bert_score.values())[0])
        avg_bert_score_recall = torch.mean(list(bert_score.values())[1])
        avg_bert_score_f1 = torch.mean(list(bert_score.values())[2])
        bert_results = [avg_bert_score_precision.item(), avg_bert_score_recall.item(), avg_bert_score_f1.item()]
        results.append("BERT: " + str(bert_results))
    return results

# Create the file for the metrics
def create_metric_file(method):
    path = './evaluation/eval_metrics_' + method + '.txt'
    if not os.path.exists(path):
        with open(path, 'w'): pass

# Write the results to the file
def average_metric_results(metric_results):
    avg_results = []
    avg_results.append(str(metric_results[0]) + "\n")
    WER_list = []
    BLEU_list = []
    ROUGE_list = []
    TER_list = []
    BERT_list = []
    for result in metric_results[1:]:
        WER_list.append(float(result[0].split("WER: ")[1]))
        BLEU_list.append(float(result[1].split("BLEU: ")[1]))
        ROUGE_list.append(float(result[2].split("ROUGE: ")[1]))
        TER_list.append(float(result[3].split("TER: ")[1]))
        bert_value = result[4].split("BERT: ")[1]
        if bert_value != 'None':
            BERT_list.append(ast.literal_eval(result[4].split("BERT: ")[1])[0])
    average_WER = sum(WER_list) / len(WER_list)
    avg_results.append("WER: " + str(average_WER))
    average_BLEU = sum(BLEU_list) / len(BLEU_list)
    avg_results.append("BLEU: " + str(average_BLEU))
    average_ROUGE = sum(ROUGE_list) / len(ROUGE_list)
    avg_results.append("ROUGE: " + str(average_ROUGE))
    average_TER = sum(TER_list) / len(TER_list)
    avg_results.append("TER: " + str(average_TER))
    average_BERT = None
    if len(BERT_list) != 0:
        average_BERT = sum(BERT_list) / len(BERT_list)
    avg_results.append("BERT: " + str(average_BERT))
    return avg_results

# Write the results to the file
def write_metric_results(results, method):
    with open(f'./evaluation/eval_metrics_{method}.txt', 'w', encoding='utf-8') as file:
        file.write(str(results) + "\n")

if __name__ == '__main__':
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
                            metric_results.append(metric_result)
                    results = average_metric_results(metric_results)
                    write_metric_results(results, method_name[i])
                    print("All metrics are calculated for " + method_name[i])