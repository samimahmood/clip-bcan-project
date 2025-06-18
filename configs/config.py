import os
import torch

class Config:
    def __init__(self):
        # Dataset
        self.dataset_name = 'flickr8k'  # Can be 'flickr8k' or 'flickr30k'
        self.data_path = '/path/to/flickr8k'  # Update this path
        
        # Auto-detect dataset structure
        if 'flickr30k' in self.dataset_name.lower():
            self.images_path = os.path.join(self.data_path, 'flickr30k-images')
            self.captions_path = os.path.join(self.data_path, 'results_20130124.token')
        else:  # flickr8k
            self.images_path = os.path.join(self.data_path, 'Flicker8k_Dataset')
            self.captions_path = os.path.join(self.data_path, 'Flickr8k.token.txt')
        
        # Model
        self.clip_model = 'ViT-B/16'
        self.embed_size = 1024  # BCAN embedding size
        self.clip_embed_size = 512  # CLIP output size
        self.dropout = 0.2
        
        # Training
        self.batch_size = 32
        self.num_epochs = 20
        self.learning_rate = 1e-4
        self.weight_decay = 0.0
        self.margin = 0.2
        self.grad_clip = 2.0
        self.log_step = 20
        self.val_step = 500
        
        # Loss weights
        self.lambda_lse = 6.0
        self.lambda_softmax = 5.0
        
        # Evaluation
        self.eval_batch_size = 128
        
        # Paths
        self.checkpoint_path = './checkpoints'
        self.log_path = './logs'
        self.best_model_path = os.path.join(self.checkpoint_path, 'best_model.pth')
        
        # Device - auto-detect best available
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'  # Apple Silicon GPU
        else:
            self.device = 'cpu'
        
        print(f"Using device: {self.device}")
        
        self.num_workers = 4
        
        # Create directories
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)