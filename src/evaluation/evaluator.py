import torch
import numpy as np
from collections import OrderedDict

class Evaluator:
    def __init__(self, config):
        self.config = config
        
    def compute_similarity(self, img_embs, txt_embs):
        """Compute similarity matrix between images and texts"""
        # Normalize embeddings
        img_embs = img_embs / img_embs.norm(dim=1, keepdim=True)
        txt_embs = txt_embs / txt_embs.norm(dim=1, keepdim=True)
        
        # Compute cosine similarity
        sim_matrix = torch.matmul(img_embs, txt_embs.t())
        
        return sim_matrix.numpy()
    
    def i2t(self, sim_matrix, return_ranks=False):
        """
        Image-to-Text retrieval
        Args:
            sim_matrix: similarity matrix of shape [n_images, n_captions]
            return_ranks: if True, return ranks
        """
        n_images = sim_matrix.shape[0]
        n_captions = sim_matrix.shape[1]
        
        # Assuming 5 captions per image
        captions_per_image = 5
        n_unique_images = n_images
        
        ranks = np.zeros(n_unique_images)
        top1 = np.zeros(n_unique_images)
        top5 = np.zeros(n_unique_images)
        top10 = np.zeros(n_unique_images)
        
        for i in range(n_unique_images):
            # Get similarity scores for this image
            sim_scores = sim_matrix[i]
            
            # Sort in descending order
            sorted_indices = np.argsort(sim_scores)[::-1]
            
            # Find the rank of the first correct caption
            # Correct captions are at indices [i*5, i*5+1, ..., i*5+4]
            correct_indices = list(range(i * captions_per_image, (i + 1) * captions_per_image))
            
            # Find minimum rank among correct captions
            min_rank = n_captions
            for idx in correct_indices:
                rank = np.where(sorted_indices == idx)[0][0]
                if rank < min_rank:
                    min_rank = rank
            
            ranks[i] = min_rank
            
            # Calculate recall metrics
            if min_rank < 1:
                top1[i] = 1
            if min_rank < 5:
                top5[i] = 1
            if min_rank < 10:
                top10[i] = 1
        
        # Compute metrics
        r1 = 100.0 * np.sum(top1) / n_unique_images
        r5 = 100.0 * np.sum(top5) / n_unique_images
        r10 = 100.0 * np.sum(top10) / n_unique_images
        
        mean_rank = ranks.mean() + 1  # Add 1 because ranks are 0-indexed
        
        results = {
            'R@1': r1,
            'R@5': r5,
            'R@10': r10,
            'mean_r': mean_rank
        }
        
        if return_ranks:
            results['ranks'] = ranks
            
        return results
    
    def t2i(self, sim_matrix, return_ranks=False):
        """
        Text-to-Image retrieval
        Args:
            sim_matrix: similarity matrix of shape [n_images, n_captions]
            return_ranks: if True, return ranks
        """
        # Transpose to get [n_captions, n_images]
        sim_matrix_t = sim_matrix.T
        
        n_captions = sim_matrix_t.shape[0]
        n_images = sim_matrix_t.shape[1]
        
        # Assuming 5 captions per image
        captions_per_image = 5
        
        ranks = np.zeros(n_captions)
        top1 = np.zeros(n_captions)
        top5 = np.zeros(n_captions)
        top10 = np.zeros(n_captions)
        
        for i in range(n_captions):
            # Get similarity scores for this caption
            sim_scores = sim_matrix_t[i]
            
            # Sort in descending order
            sorted_indices = np.argsort(sim_scores)[::-1]
            
            # Find the correct image for this caption
            correct_image_idx = i // captions_per_image
            
            # Find rank of correct image
            rank = np.where(sorted_indices == correct_image_idx)[0][0]
            ranks[i] = rank
            
            # Calculate recall metrics
            if rank < 1:
                top1[i] = 1
            if rank < 5:
                top5[i] = 1
            if rank < 10:
                top10[i] = 1
        
        # Compute metrics
        r1 = 100.0 * np.sum(top1) / n_captions
        r5 = 100.0 * np.sum(top5) / n_captions
        r10 = 100.0 * np.sum(top10) / n_captions
        
        mean_rank = ranks.mean() + 1  # Add 1 because ranks are 0-indexed
        
        results = {
            'R@1': r1,
            'R@5': r5,
            'R@10': r10,
            'mean_r': mean_rank
        }
        
        if return_ranks:
            results['ranks'] = ranks
            
        return results
    
    def evaluate(self, img_embs, txt_embs):
        """
        Evaluate retrieval performance
        Args:
            img_embs: image embeddings [n_images, embed_dim]
            txt_embs: text embeddings [n_captions, embed_dim]
        """
        # Compute similarity matrix
        sim_matrix = self.compute_similarity(img_embs, txt_embs)
        
        # Image-to-Text retrieval
        i2t_results = self.i2t(sim_matrix)
        
        # Text-to-Image retrieval
        t2i_results = self.t2i(sim_matrix)
        
        # Compile results
        results = {
            'i2t': i2t_results,
            't2i': t2i_results
        }
        
        return results