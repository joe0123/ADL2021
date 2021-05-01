import datasets
import numpy

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
            features=datasets.Features(
                {
                    "predictions": {"id": datasets.Value("string"), "pred": datasets.Value("int32")},
                    "references": {"id": datasets.Value("string"), "label": datasets.Value("int32")},
                }
            ),
        )

    def _compute(self, predictions, references):
        pred_numpy = [predictions[example_id]["pred"] for example_id in sorted(references.keys())]
        ref_numpy = [references[example_id]["label"] for example_id in sorted(references.keys())]

        score = np.mean(np.where(pred_numpy == ref_numpy, 1, 0))
        return score



