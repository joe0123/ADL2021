import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class IntentGRU(nn.Module):
    def __init__(self, embed_matrix, num_classes, args):
        super(IntentGRU, self).__init__()
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


class IntentLSTM(nn.Module):
    def __init__(self, embed_matrix, num_classes, args):
        super(IntentLSTM, self).__init__()
        self.embed = torch.nn.Embedding(embed_matrix.shape[0], embed_matrix.shape[1])
        self.embed.weight = torch.nn.Parameter(embed_matrix)
        self.embed.weight.requires_grad = False # Frozen embed layer when training

        self.embed_dim = embed_matrix.shape[1]
        self.lstm = nn.LSTM(input_size=self.embed_dim, hidden_size=args.hidden_dim, num_layers=2, dropout=args.dropout,
                batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(args.dropout),
                                        nn.Linear(in_features=args.hidden_dim * 2, out_features=args.hidden_dim * 2),
                                        nn.PReLU(),
                                        nn.Dropout(args.dropout),
                                        nn.Linear(in_features=args.hidden_dim * 2, out_features=num_classes))
    
    def forward(self, inputs):
        features, _ = self.lstm(self.embed(inputs).float(), None)
        return self.classifier(features[:, -1, :].reshape(features.shape[0], -1)).squeeze()

