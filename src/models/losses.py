import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """Contrastive loss for cross-modal retrieval"""
    def __init__(self, margin=0.2, lambda_softmax=5.0, lambda_lse=6.0):
        super().__init__()
        self.margin = margin
        self.lambda_softmax = lambda_softmax
        self.lambda_lse = lambda_lse
        
    def forward(self, img_embeds, txt_embeds):
        """
        Args:
            img_embeds: Image embeddings [batch_size, embed_dim]
            txt_embeds: Text embeddings [batch_size, embed_dim]
        """
        batch_size = img_embeds.size(0)
        
        # Compute similarity scores
        scores = torch.matmul(img_embeds, txt_embeds.t())  # [batch_size, batch_size]
        diagonal = scores.diag().view(batch_size, 1)
        
        # Create masks for positive and negative pairs
        pos_mask = torch.eye(batch_size).bool().to(scores.device)
        neg_mask = ~pos_mask
        
        # Image-to-Text loss
        d_i2t = diagonal.expand_as(scores)
        cost_i2t = (self.margin + scores - d_i2t).clamp(min=0)
        cost_i2t = cost_i2t * neg_mask.float()
        
        # Text-to-Image loss
        d_t2i = diagonal.t().expand_as(scores)
        cost_t2i = (self.margin + scores.t() - d_t2i).clamp(min=0)
        cost_t2i = cost_t2i * neg_mask.float()
        
        # Compute losses
        if self.lambda_lse > 0:
            # LogSumExp loss
            loss_i2t = torch.log(torch.sum(torch.exp(cost_i2t), dim=1) + 1e-8).mean()
            loss_t2i = torch.log(torch.sum(torch.exp(cost_t2i), dim=1) + 1e-8).mean()
            loss_lse = self.lambda_lse * (loss_i2t + loss_t2i)
        else:
            loss_lse = 0
            
        if self.lambda_softmax > 0:
            # Softmax loss
            loss_i2t_softmax = F.cross_entropy(scores, torch.arange(batch_size).to(scores.device))
            loss_t2i_softmax = F.cross_entropy(scores.t(), torch.arange(batch_size).to(scores.device))
            loss_softmax = self.lambda_softmax * (loss_i2t_softmax + loss_t2i_softmax)
        else:
            loss_softmax = 0
            
        # Total loss
        total_loss = loss_lse + loss_softmax
        
        return total_loss, {
            'loss_lse': loss_lse,
            'loss_softmax': loss_softmax,
            'loss_total': total_loss
        }


class TripletLoss(nn.Module):
    """Alternative: Triplet loss implementation"""
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
        
    def forward(self, img_embeds, txt_embeds):
        """
        Compute triplet loss
        """
        batch_size = img_embeds.size(0)
        
        # Compute all pairwise distances
        scores = torch.matmul(img_embeds, txt_embeds.t())
        
        # Get positive pairs (diagonal)
        pos_scores = scores.diag()
        
        # For each positive pair, find hardest negative
        loss_i2t = 0
        loss_t2i = 0
        
        for i in range(batch_size):
            # Image-to-text: for image i, positive text is i
            neg_scores_i2t = torch.cat([scores[i, :i], scores[i, i+1:]])
            if len(neg_scores_i2t) > 0:
                hardest_neg_i2t = neg_scores_i2t.max()
                loss_i2t += (self.margin + hardest_neg_i2t - pos_scores[i]).clamp(min=0)
            
            # Text-to-image: for text i, positive image is i
            neg_scores_t2i = torch.cat([scores[:i, i], scores[i+1:, i]])
            if len(neg_scores_t2i) > 0:
                hardest_neg_t2i = neg_scores_t2i.max()
                loss_t2i += (self.margin + hardest_neg_t2i - pos_scores[i]).clamp(min=0)
        
        total_loss = (loss_i2t + loss_t2i) / batch_size
        
        return total_loss, {
            'loss_i2t': loss_i2t / batch_size,
            'loss_t2i': loss_t2i / batch_size,
            'loss_total': total_loss
        }