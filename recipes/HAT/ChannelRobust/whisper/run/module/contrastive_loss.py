import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, Q, K):
        """
        We use InfoNCE as contrastive loss
        Args:
            Q: Tensor of shape (B, T, E), embeddings from one set
            K: Tensor of shape (B, T, E), embeddings from another set
            Key(key) is the ref model in our case.
        Returns:
            loss: Scalar tensor representing the contrastive loss
        """
        B, T, E = Q.shape

        # Normalize for the embedding dimensions to make calculations easier
        Q = F.normalize(Q, dim=2)
        K = F.normalize(K, dim=2)

        # Calculate logits for all batches at once
        logits = torch.matmul(Q, K.transpose(2, 1)) / self.temperature

        # Create labels where each positive pair is the diagonal element
        labels = torch.arange(T).long().to(Q.device)
        
        # T -> (B, T)
        labels = labels.unsqueeze(0).expand(B, -1)  

        # Calculate cross-entropy loss for each batch and take the mean
        loss = F.cross_entropy(logits.reshape(-1, T), labels.reshape(-1), reduction='mean')

        return loss

class RandomContrastiveLoss(ContrastiveLoss):
    def __init__(self, temperature=0.07, num_of_embs=300):
        super(RandomContrastiveLoss, self).__init__(temperature)
        self.num_of_embs = num_of_embs
    
    def forward(self, Q, K):
        B, T, E = Q.shape
        
        # Normalize for the embedding dimensions to make calculations easier
        Q = F.normalize(Q, dim=2)
        K = F.normalize(K, dim=2)
        
        # Select random timestamps
        idx = torch.randperm(T)[:self.num_of_embs]
        Q = Q[:, idx, :]
        K = K[:, idx, :]
        
        # Calculate logits for all batches at once
        logits = torch.matmul(Q, K.transpose(2, 1)) / self.temperature
        
        # Create labels where each positive pair is the diagonal element
        labels = torch.arange(self.num_of_embs).long().to(Q.device)
        labels = labels.unsqueeze(0).expand(B, -1)  # (B, num_of_embs)
        
        # Calculate cross-entropy loss for each batch and take the mean
        loss = F.cross_entropy(logits.reshape(-1, self.num_of_embs), labels.reshape(-1), reduction='mean')
        
        return loss

class PatchedContrastiveLoss(ContrastiveLoss):
    def __init__(self, temperature=0.07, patch_size=300):
        super(PatchedContrastiveLoss, self).__init__(temperature)
        self.patch_size = patch_size
    
    def forward(self, Q, K):
        B, T, E = Q.shape
        
        # Calculate the new size of T after patching
        new_T = T // self.patch_size
        new_E = E * self.patch_size
        
        # Normalize for the embedding dimensions to make calculations easier
        Q = F.normalize(Q, dim=2)
        K = F.normalize(K, dim=2)
        
        # Reshape Q and K to form patches
        Q = Q.view(B, new_T, new_E)
        K = K.view(B, new_T, new_E)

        
        # Calculate logits for all batches at once
        logits = torch.matmul(Q, K.transpose(2, 1)) / self.temperature
        
        # Create labels where each positive pair is the diagonal element
        labels = torch.arange(new_T).long().to(Q.device)
        labels = labels.unsqueeze(0).expand(B, -1)  # (B, new_T)
        
        # Calculate cross-entropy loss for each batch and take the mean
        loss = F.cross_entropy(logits.reshape(-1, new_T), labels.reshape(-1), reduction='mean')
        
        return loss


class AdaptiveSoftContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, neighbor_num=10):
        super(AdaptiveSoftContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.neighbor_num = neighbor_num

    def forward(self, Q, K):
        B, T, E = Q.shape
        Q = F.normalize(Q, dim=2)
        K = F.normalize(K, dim=2)
        
        loss = torch.zeros(B, device=Q.device, dtype=torch.float64)
        
        for i in range(B):
            q = Q[i]
            k = K[i]
            
            dist_k = torch.matmul(k, k.T)
            mask = ~torch.eye(T, dtype=bool, device=q.device)
            logits_pd = dist_k[mask].reshape(T, -1)
            
            pseudo_labels = F.softmax(logits_pd / self.temperature, dim=1)
            log_pseudo_labels = F.log_softmax(logits_pd / self.temperature, dim=1)
            
            entropy = -torch.sum(pseudo_labels * log_pseudo_labels, dim=1, keepdim=True)
            max_entropy = np.log(T - 1)
            
            c = 1 - entropy / max_entropy
            pseudo_labels = self.neighbor_num * c * pseudo_labels
            pseudo_labels = torch.minimum(pseudo_labels, torch.tensor(1.0, device=pseudo_labels.device))
            
            labels = torch.zeros_like(dist_k, dtype=pseudo_labels.dtype)
            labels.fill_diagonal_(1)
            labels[mask] = pseudo_labels.reshape(-1)
            labels = labels / labels.sum(dim=1, keepdim=True)
            
            dist_real = torch.matmul(q, k.T)
            loss[i] = (2 - 2 * (labels * dist_real).sum(dim=-1)).mean()
        
        return loss.mean()

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, p, q):
        """
        Args:
            p: Tensor of shape (B, T, E)
            q: Tensor of shape (B, T, E)
        
        Returns:
            loss: Mean Squared Error loss value
        """
        return torch.sum(torch.mean((p - q) ** 2, dim=[1, 2]))