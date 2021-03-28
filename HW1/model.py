import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class IntentGRU(nn.Module):
    def __init__(self, num_classes, pad_class, args):
        super(IntentGRU, self).__init__()
        assert self.cri_name == "ce", "Criterion must be cross entropy loss in task intent!"
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

    def predict(self, inputs, input_lens):
        outputs = self.forward(inputs, input_lens)
        return torch.argmax(outputs, dim=-1).int()


class SlotGRU(nn.Module):
    def __init__(self, num_classes, pad_class, args):
        super(SlotGRU, self).__init__()
        self.args = args
        if args.cri_name == "crf":
            self.critesion = args.criterion(num_classes, batch_first=True)
        else:
            self.criterion = args.criterion(reduction="sum")
        self.num_classes = num_classes
        self.pad_class = pad_class

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
        return self.classifier(features)

    def compute_loss(self, inputs, input_lens, targets):
        pad_mask = (torch.arange(0, inputs.shape[1]).repeat(inputs.shape[0], 1).to(input_lens.device) \
                        >= input_lens.unsqueeze(1)).float()
        outputs = self.forward(inputs, input_lens)
        if self.args.cri_name == "crf":
            return self.criterion(outputs, targets, 1 - pad_mask, reduction="sum") / input_lens.sum()
        else:
            outputs[:, :, self.pad_class] += 1e+8 * pad_mask
            return self.criterion(outputs.reshape(-1, self.num_classes), targets.reshape(-1)) / input_lens.sum()

    def predict(self, inputs, input_lens):
        pad_mask = (torch.arange(0, inputs.shape[1]).repeat(inputs.shape[0], 1).to(input_lens.device) \
                        >= input_lens.unsqueeze(1)).float()
        outputs = self.forward(inputs, input_lens)
        if self.args.cri_name == "crf":
            return self.criterion.decode(outputs, 1 - pad_mask)
        else:
            outputs[:, :, self.pad_class] += 1e+8 * pad_mask
            return torch.argmax(outputs, dim=-1).int()

