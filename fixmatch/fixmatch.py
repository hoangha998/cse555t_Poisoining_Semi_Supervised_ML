import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class FixMatch(nn.Module):
    def __init__(self, model, T=0.5):
        super().__init__()
        self.model = model
        self.T = T

    def forward(self, X_u, y_u, X_l, y_l, X_ul, unlabeled_mask, labeled_mask):
        self.model.train()
        
        # Get predictions on unlabeled data
        logits_ul = self.model(X_ul)
        probs_ul = F.softmax(logits_ul / self.T, dim=1)
        preds_ul = torch.argmax(probs_ul, dim=1)
        
        # Create pseudo-labels for unlabeled data
        y_ul = preds_ul.detach()
        y_ul[unlabeled_mask == 0] = -1  # mask out padding values
        
        # Combine labeled and pseudo-labeled data
        X = torch.cat([X_u, X_ul], dim=0)
        y = torch.cat([y_u, y_ul], dim=0)
        
        # Train the model on combined data with labeled and pseudo-labels
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        optimizer.zero_grad()
        logits = self.model(X)
        loss = F.cross_entropy(logits[labeled_mask], y[labeled_mask])
        loss.backward()
        optimizer.step()
        
        return loss.item()
'''