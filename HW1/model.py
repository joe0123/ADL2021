import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class IntentGRU(nn.Module):
    def __init__(self, num_classes, pad_class, args):
        super(IntentGRU, self).__init__()
        assert args.cri_name == "ce", "Criterion must be cross entropy loss in task intent!"
        if hasattr(args, "criterion"):
            self.criterion = args.criterion(reduction="sum")

        embed_matrix = torch.load(os.path.join(args.cache_dir, "embeddings.pt"))
        self.embed = torch.nn.Embedding(embed_matrix.shape[0], embed_matrix.shape[1])
        self.embed.weight = torch.nn.Parameter(embed_matrix)
        self.embed.weight.requires_grad = False # Frozen embed layer when training

        self.embed_dim = embed_matrix.shape[1]
        self.gru = nn.GRU(input_size=self.embed_dim, hidden_size=args.hidden_dim, num_layers=2, dropout=args.dropout,
                batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(args.dropout),
                                        nn.Linear(in_features=args.hidden_dim * 2, out_features=args.hidden_dim * 2),
                                        nn.PReLU(),
                                        nn.Dropout(args.dropout),
                                        nn.Linear(in_features=args.hidden_dim * 2, out_features=num_classes))
        self.pack = lambda inputs, input_lens: pack_padded_sequence(inputs, input_lens, \
                                                            batch_first=True, enforce_sorted=False)
        self.unpack = lambda inputs: pad_packed_sequence(inputs, batch_first=True)
        
            
    def forward(self, inputs, input_lens):
        packed_features, _ = self.gru(self.pack(self.embed(inputs).float(), input_lens.cpu()), None)
        features, input_lens_ = self.unpack(packed_features)
        assert torch.all(torch.eq(input_lens.cpu(), input_lens_.cpu()))
        selected = (input_lens - 1).unsqueeze(-1).repeat(1, features.shape[-1]).unsqueeze(1)
        return self.classifier(torch.gather(features, dim=1, index=selected).reshape(features.shape[0], -1)).squeeze()

    def compute_loss(self, inputs, input_lens, targets):
        outputs = self.forward(inputs, input_lens)
        return self.criterion(outputs, targets) / inputs.shape[0]
    
    def score(self, inputs, input_lens, targets, reduction="sum"):
        preds = self.predict(inputs, input_lens)
        acc = sum([1 if pred == target else 0 for pred, target in zip(preds, targets)])
        if reduction == "sum":
            return acc
        elif reduction == "mean":
            return acc / len(preds)

    def predict(self, inputs, input_lens):
        outputs = self.forward(inputs, input_lens)
        return torch.argmax(outputs, dim=-1).int().tolist()


class SlotGRU(nn.Module):
    def __init__(self, num_classes, pad_class, args):
        super(SlotGRU, self).__init__()
        self.args = args
        if hasattr(args, "criterion"):
            if args.cri_name == "crf":
                self.criterion = args.criterion(num_classes, batch_first=True)
            else:
                self.criterion = args.criterion(reduction="sum")
        self.num_classes = num_classes
        self.pad_class = pad_class

        embed_matrix = torch.load(os.path.join(args.cache_dir, "embeddings.pt"))
        self.embed = torch.nn.Embedding(embed_matrix.shape[0], embed_matrix.shape[1])
        self.embed.weight = torch.nn.Parameter(embed_matrix)
        self.embed.weight.requires_grad = False # Frozen embed layer when training
        self.num_vocabs, self.embed_dim = embed_matrix.shape[0], embed_matrix.shape[1]

        self.gru = nn.GRU(input_size=self.embed_dim, hidden_size=args.hidden_dim, num_layers=2, dropout=args.dropout,
                batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(args.dropout),
                                        nn.Linear(in_features=args.hidden_dim * 2, out_features=args.hidden_dim * 2),
                                        nn.PReLU(),
                                        nn.Dropout(args.dropout),
                                        nn.Linear(in_features=args.hidden_dim * 2, out_features=num_classes))
        
        
        self.pack = lambda inputs, input_lens: pack_padded_sequence(inputs, input_lens, \
                                                            batch_first=True, enforce_sorted=False)
        self.unpack = lambda inputs: pad_packed_sequence(inputs, batch_first=True)
         
         
    def get_emissions(self, inputs, input_lens):
        embeds = self.embed(inputs).float()
        packed_features, _ = self.gru(self.pack(embeds, input_lens.cpu()), None)
        features, input_lens_ = self.unpack(packed_features)
        assert torch.all(torch.eq(input_lens.cpu(), input_lens_.cpu()))
        return features

    def compute_loss(self, inputs, input_lens, targets):
        pad_mask = (torch.arange(0, inputs.shape[1]).repeat(inputs.shape[0], 1).to(input_lens.device) \
                        >= input_lens.unsqueeze(1))
        features = self.get_emissions(inputs, input_lens)
        outputs = self.classifier(features)
        if self.args.cri_name == "crf":
            loss = -self.criterion(outputs, targets, torch.logical_not(pad_mask), reduction="sum") / input_lens.sum()
        else:
            outputs[:, :, self.pad_class] += 1e+8 * pad_mask.float()
            loss = self.criterion(outputs.reshape(-1, self.num_classes), targets.reshape(-1)) / input_lens.sum()
        
        return loss
    
    def score(self, inputs, input_lens, targets, reduction="sum"):
        preds = self.predict(inputs, input_lens)
        acc = sum([1 if pred == target else 0 for pred, target in zip(preds, targets)])
        if reduction == "sum":
            return acc
        elif reduction == "mean":
            return acc / len(preds)

    def predict(self, inputs, input_lens):
        pad_mask = (torch.arange(0, inputs.shape[1]).repeat(inputs.shape[0], 1).to(input_lens.device) \
                        >= input_lens.unsqueeze(1))
        outputs = self.classifier(self.get_emissions(inputs, input_lens))
        if self.args.cri_name == "crf":
            preds = self.criterion.decode(outputs, torch.logical_not(pad_mask))
        else:
            outputs[:, :, self.pad_class] += 1e+8 * pad_mask.float()
            preds = torch.argmax(outputs, dim=-1).int().cpu().tolist()
            preds = [pred[:input_len] for pred, input_len in zip(preds, input_lens.cpu().tolist())]
        return preds
