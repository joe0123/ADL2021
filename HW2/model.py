import torch
from torch import nn

from transformers import BertPreTrainedModel, BertModel

class QABert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        
        self.rel_extractor = nn.Sequential(
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Tanh(),
        )
        
        self.rel_classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 1),
        )

        self.qa_classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        rel_labels=None,
        start_labels=None,
        end_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):
        output_hidden_states=True   # For rel extractor
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        rel_embed = torch.cat([t.unsqueeze(0) for t in outputs["hidden_states"][-4:]], dim=0)[:, :, 0: 1, :]
        rel_embed = rel_embed.permute(1, 2, 0, 3)
        rel_embed = self.rel_extractor(rel_embed.reshape(rel_embed.shape[0], -1))
        rel_logits = self.rel_classifier(rel_embed).squeeze(-1)

        qa_logits = self.qa_classifier(outputs["last_hidden_state"])
        start_logits, end_logits = qa_logits[:, :, 0], qa_logits[:, :, 1]
        # TODO mask type_ids=0 or not
        
        rel_loss = None
        if rel_labels is not None:
            if len(rel_labels.shape) > 1:
                rel_labels = rel_labels.squeeze(-1)
            rel_criterion = nn.BCEWithLogitsLoss(reduction="sum")
            rel_loss = rel_criterion(rel_logits, rel_labels)

        qa_loss = None
        if start_labels is not None and end_labels is not None:
            if len(start_labels.shape) > 1:
                start_labels = start_labels.squeeze(-1)
            if len(end_labels.shape) > 1:
                end_labels = end_labels.squeeze(-1)
            
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_labels.clamp_(0, ignored_index)
            end_labels.clamp_(0, ignored_index)

            qa_criterion = nn.CrossEntropyLoss(ignore_index=ignored_index, reduction="sum")
            start_loss = qa_criterion(start_logits, start_labels)
            end_loss = qa_criterion(end_logits, end_labels)
            qa_loss = (start_loss + end_loss) / 2

        if not return_dict:
            outputs = (rel_logits, start_logits, end_logits)
            return ((rel_loss, qa_loss,) + outputs) if rel_loss is not None and qa_loss is not None else outputs
        else:
            outputs = {"rel_loss": rel_loss,
                        "qa_loss": qa_loss,
                        "rel_logits": rel_logits,
                        "start_logits": start_logits,
                        "end_logits": end_logits
            }
            return outputs
