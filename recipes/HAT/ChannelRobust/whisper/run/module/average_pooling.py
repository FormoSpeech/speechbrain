import torch, torch.nn as nn

class AveragePooling(nn.Module):
    def __init__(self):
        super(AveragePooling, self).__init__()
    def forward(self, x):        
        # x is expected to be of shape (B, T, E)
        # Perform mean pooling along the sequence length (T) dimension
        pooled = torch.mean(x, dim=1)
        
        # Project the pooled tensor to the desired output dimension
        # projected = self.projection(pooled)  # Resulting shape is (B, output_dim)
        return pooled