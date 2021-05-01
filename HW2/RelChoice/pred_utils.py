import collections
import json
import logging
import os

import numpy as np
from transformers import EvalPrediction
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)

def post_processing_function(examples, features, pred_logits, args):
    predictions = postprocess_relchoice_predictions(
        examples=examples,
        features=features,
        pred_logits=pred_logits,
    )

    formatted_predictions = [{"id": k, "pred": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "label": ex[args.label_col]} for ex in examples]

    return EvalPrediction(predictions=formatted_predictions, label_ids=references)

def postprocess_relchoice_predictions(
    examples,
    features,
    pred_logits,
    is_world_process_zero: bool = True,
):

    assert pred_logits[0].shape[0] == features.shape[0], \
            f"Got {len(pred_logits[0])} pred_logits and {len(features)} features."

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    all_predictions = collections.OrderedDict()

    logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")
    
    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]

        prelim_predictions = []
        for feature_index in feature_indices:
            prelim_predictions.append(pred_logits[feature_index])
        prediction = np.argmax(prelim_predictions)
        
        all_predictions[example["id"]] = prediction 
    
    return all_predictions


