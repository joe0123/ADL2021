import collections
import json
import logging
import os
from typing import Optional, Tuple

import numpy as np
from transformers import EvalPrediction
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)

def post_processing_function(examples, features, predictions, args, model):
    if args.beam:
        predictions = postprocess_qa_predictions_with_beam_search(
            examples=examples,
            features=features,
            predictions=predictions,
            n_best=args.n_best,
            max_ans_len=args.max_ans_len,
            start_n_top=model.config.start_n_top,
            end_n_top=model.config.end_n_top,
        )
    else:
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            n_best=args.n_best,
            max_ans_len=args.max_ans_len,
        )

    formatted_predictions = [{"id": k, "pred": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex[args.ans_col]} for ex in examples]

    return EvalPrediction(predictions=formatted_predictions, label_ids=references)

def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    n_best: int = 20,
    max_ans_len: int = 30,
    is_world_process_zero: bool = True,
):
    assert len(predictions) == 2, "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = predictions

    assert predictions[0].shape[0] == features.shape[0], \
            f"Got {len(predictions[0])} predictions and {len(features)} features."

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")
    
    # TODO: More robust: add null score?
    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]

        prelim_predictions = []
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            start_indexes = np.argsort(start_logits)[-1: -n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logits)[-1: -n_best - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (start_index >= len(offset_mapping) or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None or offset_mapping[end_index] is None):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_ans_len:
                        continue
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best]
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0]: offsets[1]]
        
        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ''):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        all_predictions[example["id"]] = predictions[0]["text"]
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]
    
    return all_predictions


def postprocess_qa_predictions_with_beam_search(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    n_best: int = 20,
    max_ans_len: int = 30,
    start_n_top: int = 5,
    end_n_top: int = 5,
    is_world_process_zero: bool = True,
):
    assert len(predictions) == 5, "`predictions` should be a tuple with five elements."
    start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits = predictions

    assert predictions[0].shape[0] == features.shape[0], \
            f"Got {predictions[0].shape[0]} predicitions and {features.shape[0]} features."

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    for example_index, example in enumerate(tqdm(examples)):
        feature_indices = features_per_example[example_index]

        prelim_predictions = []
        for feature_index in feature_indices:
            start_log_prob = start_top_log_probs[feature_index]
            start_indexes = start_top_index[feature_index]
            end_log_prob = end_top_log_probs[feature_index]
            end_indexes = end_top_index[feature_index]
            feature_null_score = cls_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_index = int(start_indexes[i])
                    j_ = i * end_n_top + j
                    end_index = int(end_indexes[j_])
                    if (start_index >= len(offset_mapping) or end_index >= len(offset_mapping) \
                        or offset_mapping[start_index] is None or offset_mapping[end_index] is None):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_ans_len:
                        continue
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_log_prob[i] + end_log_prob[j_],
                            "start_log_prob": start_log_prob[i],
                            "end_log_prob": end_log_prob[j_]
                        }
                    )

        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best]
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0]: offsets[1]]

        if len(predictions) == 0:
            predictions.insert(0, {"text": '', "start_logit": -1e-6, "end_logit": -1e-6, "score": -2e-6})

        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        all_predictions[example["id"]] = predictions[0]["text"]
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]

    return all_predictions


def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float32)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]
            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat

