import datasets
import numpy as np

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
        pred_dict = {prediction["id"]: prediction["pred"] for prediction in predictions}
        ref_dict = {reference["id"]: reference["label"] for reference in references}
        example_ids = ref_dict.keys()
        pred_numpy = np.array([pred_dict[example_id] for example_id in example_ids])
        ref_numpy = np.array([ref_dict[example_id] for example_id in example_ids])

        score = np.mean(np.where(pred_numpy == ref_numpy, 1, 0))
        return score



