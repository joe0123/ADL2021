import sys
import datasets
import collections
import json
import os

import spacy
from tqdm import tqdm

_CITATION = ''
_DESCRIPTION = ''
_KWARGS_DESCRIPTION = ''

@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class EM_F1(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": {"id": datasets.Value("string"), "pred": datasets.Value("string")},
                    "references": {
                        "id": datasets.Value("string"),
                        "answers": datasets.features.Sequence(
                            {
                                "text": datasets.Value("string"),
                                "start": datasets.Value("int32"),
                            }
                        ),
                    },
                }
            ),
        )

    def _compute(self, predictions, references):
        pred_dict = {prediction["id"]: prediction["pred"] for prediction in predictions}
        ref_dict = {reference["id"]: {"answers": reference["answers"]["text"]} for reference in references}
        tokenizer = Tokenizer()

        score = compute_metrics(ref_dict, pred_dict, tokenizer)
        return score



class Tokenizer:
    def __init__(self):
        self.nlp = spacy.load('zh_core_web_md', disable=['ner', 'parser', 'tagger'])

    def __call__(self, text, remove_punc=False):
        tokens = list(self.nlp(text))
        if remove_punc:
            tokens = [e for e in tokens if not e.is_punct]
        tokens = [e.text for e in tokens]
        return tokens


def compute_em(ans, pred):
    def em(a, p):
        return int(''.join(a) == ''.join(p))

    return max([em(a, pred) for a in ans])


def compute_f1(ans, pred):
    def f1(a, p):
        common = collections.Counter(a) & collections.Counter(p)
        tp = sum(common.values())
        if tp == 0:
            return 0
        precision = tp / len(p)
        recall = tp / len(a)

        return (2 * precision * recall) / (precision + recall)

    return max([f1(a, pred) for a in ans])


def compute_metric(ans, pred, tokenizer):
    ans = [tokenizer(a, remove_punc=True) for a in ans]
    pred = tokenizer(pred, remove_punc=True)

    return {
        'em': compute_em(ans, pred),
        'f1': compute_f1(ans, pred)
    }


def compute_metrics(answers, predictions, tokenizer):
    metrics = []
    for id_ in tqdm(list(answers.keys()), desc='[*] Evaluating', dynamic_ncols=True):
        if id_ not in predictions:
            print(f'[!] Cannot find answer for id {id_} in model predictions')
            continue
        prediction = predictions[id_]
        metric = compute_metric(answers[id_]['answers'], prediction, tokenizer)
        metrics.append(metric)
        #if metric["em"] < 1 or metric["f1"] < 1:
            #print(id_, answers[id_]['answers'], prediction, metric["em"], metric["f1"])

    n_total = len(metrics)
    result = {
        'count': n_total,
        'em': sum([m['em'] for m in metrics]) / n_total,
        'f1': sum([m['f1'] for m in metrics]) / n_total
    }

    return result


