from pytorch_revgrad import RevGrad
import torch, torch.nn as nn

class GradientReversalLayer(nn.Module):
    
    '''
        this component is used for whole block layer embeddings to do attentive pooling, the default setting is for whisper-tiny
        output 256 dim
    '''
    def __init__(self, grl_weight = 0.5):
        super().__init__()
        self.grl = RevGrad(alpha = grl_weight)
        
    def forward(self, x):
        x = self.grl(x)
        return x