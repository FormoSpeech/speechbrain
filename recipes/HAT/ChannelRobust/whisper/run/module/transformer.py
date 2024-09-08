import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, num_layers=6):  # Specify the number of layers
        super(Transformer, self).__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=384, nhead=8, dim_feedforward=512, dropout=0.1)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x shape: [batch_size, sequence_length, feature_dim]
        x = self.transformer(x)
        return x