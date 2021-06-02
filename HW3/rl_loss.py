import torch
from torch.nn import CrossEntropyLoss
import numpy as np

from metrics import *

def compute_rl_loss(data, logits, refs, model, tokenizer, accelerator, args):
    baseline_tokens = accelerator.unwrap_model(model).generate(
        data["input_ids"],
        attention_mask=data["attention_mask"],
        max_length=args.max_target_len,
    )
    baseline_tokens = accelerator.pad_across_processes(
        baseline_tokens, dim=1, pad_index=tokenizer.pad_token_id
    )
    baseline_tokens = accelerator.gather(baseline_tokens)
    baseline_preds = tokenizer.batch_decode(baseline_tokens.cpu().numpy(), skip_special_tokens=True)

    sampled_tokens = accelerator.unwrap_model(model).generate(
        data["input_ids"],
        attention_mask=data["attention_mask"],
        max_length=args.max_target_len,
        do_sample=True,
        top_k=args.rl_top_k,
        top_p=args.rl_top_p,
        temperature=args.rl_temperature
    )
    sampled_tokens = accelerator.pad_across_processes(
        sampled_tokens, dim=1, pad_index=tokenizer.pad_token_id
    )
    sampled_tokens = accelerator.gather(sampled_tokens)
    sampled_preds = tokenizer.batch_decode(sampled_tokens.cpu().numpy(), skip_special_tokens=True)
    
    baseline_scores = compute_rouge(predictions=baseline_preds, references=refs, avg=False)
    baseline_rewards = [(scores["rouge-1"]['f'] + scores["rouge-2"]['f'] + scores["rouge-l"]['f']) / 3 \
                        for scores in baseline_scores]
    sampled_scores = compute_rouge(predictions=sampled_preds, references=refs, avg=False)
    sampled_rewards = [(scores["rouge-1"]['f'] + scores["rouge-2"]['f'] + scores["rouge-l"]['f']) / 3 \
                        for scores in sampled_scores]
    
    loss_fct = CrossEntropyLoss(reduction="none")
    loss_input = logits[:, :sampled_tokens.shape[1], :].reshape(-1, logits.shape[-1])
    loss_target = sampled_tokens.reshape(-1)
    sampled_probs = -loss_fct(loss_input, loss_target).reshape(logits.shape[0], -1).sum(1)
    diff_rewards = (torch.Tensor(baseline_rewards) - torch.Tensor(sampled_rewards)).to(sampled_probs.device)
    rl_loss = (diff_rewards * sampled_probs).mean()
    
    return rl_loss

