import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, embed_matrix, args):
        super(LSTM, self).__init__()
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
                                        nn.Linear(in_features=args.hidden_dim * 2, out_features=1),
                                        nn.Sigmoid())
    
    def forward(self, inputs):
        features, _ = self.lstm(self.embed(inputs).float(), None)
        return self.classifier(features[:, -1, :].reshape(features.shape[0], -1)).squeeze()

