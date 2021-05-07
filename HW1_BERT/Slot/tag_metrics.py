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
class Accuracy(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features({
                "predictions": datasets.Value("null"),
                "references": datasets.Value("null"),
            }),
        )

    def _compute(self, predictions, references):
        predictions = np.array(predictions)
        references = np.array(references)
        targets = np.where(references != -100)
        score = np.all(predictions[targets] == references[targets], axis=1)
        print(score)
        score = np.mean(score)
        return score


