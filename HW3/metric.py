import datasets
from tw_rouge import get_rouge


_CITATION = ''
_DESCRIPTION = ''
_KWARGS_DESCRIPTION = ''

@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class TWRouge(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                    "references": datasets.Value("string")
                }
            ),
        )

    def _compute(self, predictions, references):
        predictions = [pred.strip() + '\n' for pred in predictions]
        references = [ref.strip() + '\n' for ref in references]
        return get_rouge(predictions, references)



