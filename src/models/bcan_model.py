import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ProjectionLayer(nn.Module):
    """Project CLIP embeddings to BCAN embedding space"""
    def __init__(self, input_dim, output_dim, dropout=0.2):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class LocalCorrectUnit(nn.Module):
    """Local Correct Unit (LCU) for fine-grained alignment"""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Attention layers
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, query, key, value=None):
        """
        Args:
            query: [batch_size, embed_dim]
            key: [batch_size, embed_dim]
            value: [batch_size, embed_dim] (if None, use key)
        """
        if value is None:
            value = key
            
        # Project inputs
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.embed_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)
        
        # Output projection and residual connection
        output = self.out_proj(attended)
        output = self.layer_norm(output + query)
        
        return output


class GlobalCorrectUnit(nn.Module):
    """Global Correct Unit (GCU) for overall alignment"""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Global feature extraction
        self.global_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.Sigmoid()
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, embed_dim]
        """
        # Compute global correction weights
        weights = self.global_fc(x)
        
        # Apply correction
        corrected = x * weights
        output = self.layer_norm(corrected + x)
        
        return output


class BCAN(nn.Module):
    """Bidirectional Correct Attention Network"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Projection layers for CLIP embeddings
        self.image_proj = ProjectionLayer(
            config.clip_embed_size, 
            config.embed_size,
            config.dropout
        )
        self.text_proj = ProjectionLayer(
            config.clip_embed_size,
            config.embed_size,
            config.dropout
        )
        
        # Local Correct Units
        self.lcu_i2t = LocalCorrectUnit(config.embed_size)
        self.lcu_t2i = LocalCorrectUnit(config.embed_size)
        
        # Global Correct Units
        self.gcu_image = GlobalCorrectUnit(config.embed_size)
        self.gcu_text = GlobalCorrectUnit(config.embed_size)
        
        # Final normalization
        self.final_layer_norm = nn.LayerNorm(config.embed_size)
        
    def forward_image(self, image_features):
        """Forward pass for image features"""
        # Project CLIP features
        img_embed = self.image_proj(image_features)
        
        # Apply Global Correct Unit
        img_embed = self.gcu_image(img_embed)
        
        # Normalize
        img_embed = self.final_layer_norm(img_embed)
        img_embed = F.normalize(img_embed, p=2, dim=1)
        
        return img_embed
    
    def forward_text(self, text_features):
        """Forward pass for text features"""
        # Project CLIP features
        txt_embed = self.text_proj(text_features)
        
        # Apply Global Correct Unit
        txt_embed = self.gcu_text(txt_embed)
        
        # Normalize
        txt_embed = self.final_layer_norm(txt_embed)
        txt_embed = F.normalize(txt_embed, p=2, dim=1)
        
        return txt_embed
    
    def forward(self, image_features, text_features):
        """
        Forward pass with bidirectional attention
        Args:
            image_features: CLIP image features [batch_size, clip_embed_size]
            text_features: CLIP text features [batch_size, clip_embed_size]
        """
        # Get base embeddings
        img_embed = self.forward_image(image_features)
        txt_embed = self.forward_text(text_features)
        
        # Apply Local Correct Units (bidirectional)
        img_corrected = self.lcu_i2t(img_embed, txt_embed)
        txt_corrected = self.lcu_t2i(txt_embed, img_embed)
        
        # Final normalization
        img_final = F.normalize(img_corrected, p=2, dim=1)
        txt_final = F.normalize(txt_corrected, p=2, dim=1)
        
        return img_final, txt_final
    
    def compute_similarity(self, img_embeds, txt_embeds):
        """Compute similarity matrix between images and texts"""
        # Cosine similarity
        sim_matrix = torch.matmul(img_embeds, txt_embeds.t())
        return sim_matrix