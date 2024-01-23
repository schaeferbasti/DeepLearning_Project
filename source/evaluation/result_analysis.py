from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

matrix = []
for path in Path.cwd().iterdir():
    if "result_analysis" not in str(path) and "__init__.py" not in str(path) and "autoencoders" not in str(path) and "Combination" not in str(path) and "T5" not in str(path) and "__pycache__" not in str(path):
        evaluation_path = path.joinpath("code", "results", "evaluation")
        for eval_path in evaluation_path.iterdir():
            if "test_bleu_score" not in str(eval_path):
                with open(eval_path, 'r') as f:
                    row = []
                    content = f.read()
                    data = content.split("WER: ")[2]
                    wer = data.split("', 'BLEU: ")[0]
                    row.append(float(wer))
                    data = content.split("BLEU: ")[2]
                    bleu = data.split("', 'ROUGE: ")[0]
                    row.append(float(bleu))
                    data = content.split("ROUGE: ")[1]
                    rouge = data.split("', 'TER: ")[0]
                    row.append(float(rouge))
                    data = content.split("TER: ")[2]
                    ter = data.split("', 'BERT: ")[0]
                    row.append(float(ter))
                    data = content.split("BERT: ")[2]
                    bert = data.split("']")[0]
                    row.append(float(bert))
                    matrix.append(row)
print(matrix)

df = pd.DataFrame(matrix)
columns = ["WER", "BLEU", "ROUGE", "WER", "BERT"]
df.columns = columns

fig, ax1 = plt.subplots()
im = ax1.matshow(df.corr())
# Set custom tick labels
tick_labels = columns
ax1.set_xticks(range(len(tick_labels)))
ax1.set_xticklabels(tick_labels)
ax1.set_yticks(range(len(tick_labels)))
ax1.set_yticklabels(tick_labels)
 # Display exact values in each field
for i in range(len(tick_labels)):
    for j in range(len(tick_labels)):
        text = f'{df.corr().iloc[i, j]:.2f}'
        ax1.text(j, i, text, ha='center', va='center', color='w')
# Legend
fig.colorbar(im, ax=ax1)
#Title
ax1.set(title='Correlation Matrix of the Metrics')
plt.show()