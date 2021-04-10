import json
from seqeval.metrics import classification_report, accuracy_score
from seqeval.scheme import IOB2

eval_data = json.load(open("./data/slot/eval.json", 'r'))
eval_labels = [i["tags"] for i in sorted(eval_data, key=lambda i: i["id"])]
with open("result_slot2.csv", 'r') as f:
    pred_labels = [line.strip().split(',')[1].split() for line in f.readlines()[1:]]
print(classification_report(eval_labels, pred_labels, mode="strict", scheme=IOB2))
print("Joint Acc:", sum([1 if true == pred else 0 for true, pred in zip(eval_labels, pred_labels)]) / len(eval_labels))
print("Token Acc:", accuracy_score(eval_labels, pred_labels))
