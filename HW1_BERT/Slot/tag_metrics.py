import datasets


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
                "predictions": datasets.Sequence(datasets.Value("int32", id="label"), id="sequence"),
                "references": datasets.Sequence(datasets.Value("int32", id="label"), id="sequence"),
                }),
        )

    def _compute(self, predictions, references):
        trues = [p == r for p, r in zip(predictions, references)]
        score = sum(trues) / len(trues)
        return score


