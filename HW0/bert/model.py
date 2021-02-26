import torch
import torch.nn as nn

class BERT_based(nn.Module):
    def __init__(self, bert, args):
        super(BERT_based, self).__init__()
        self.bert = bert
        
        self.extractor = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(args.bert_dim * 4, args.bert_dim),
            nn.Tanh(),
        )
        
        self.classifier = nn.Sequential(
                nn.Dropout(args.dropout),
                nn.Linear(args.bert_dim, args.class_num),
                nn.Sigmoid()
        )

    def forward(self, inputs):
        bert_text_indices, bert_mask_indices = inputs[0], inputs[1]
        outputs = self.bert(input_ids=bert_text_indices, attention_mask=bert_mask_indices, output_hidden_states=True)
        
        embed = torch.cat([t.unsqueeze(0) for t in outputs.hidden_states[-4:]], dim=0)[:, :, 0: 1, :].permute(1, 2, 0, 3)
        embed = self.extractor(embed.reshape(embed.shape[0], -1))
        logits = self.classifier(embed).squeeze()

        return logits
