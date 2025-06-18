import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import clip
import json
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bcan_model import BCAN
from src.models.losses import ContrastiveLoss
from src.data.flickr30k_dataset import FlickrDataLoader
from src.evaluation.evaluator import Evaluator
from src.utils.logger import Logger

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize logger
        self.logger = Logger(config.log_path)
        
        # Load data
        print(f"Loading {config.dataset_name} data...")
        data_loader = FlickrDataLoader(config)
        self.train_loader, self.val_loader, self.test_loader, self.clip_model = data_loader.get_loaders()
        
        # Freeze CLIP model
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Initialize BCAN model
        print("Initializing BCAN model...")
        self.model = BCAN(config).to(self.device)
        
        # Initialize loss
        self.criterion = ContrastiveLoss(
            margin=config.margin,
            lambda_softmax=config.lambda_softmax,
            lambda_lse=config.lambda_lse
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize evaluator
        self.evaluator = Evaluator(config)
        
        # Training stats
        self.best_rsum = 0
        self.start_epoch = 0
        
    def extract_clip_features(self, images, captions):
        """Extract features using CLIP model"""
        with torch.no_grad():
            # Extract image features
            image_features = self.clip_model.encode_image(images)
            
            # Extract text features
            text_features = self.clip_model.encode_text(captions)
            
        return image_features, text_features
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        n_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch}') as pbar:
            for i, batch in enumerate(pbar):
                # Move data to device
                images = batch['image'].to(self.device)
                captions = batch['caption'].to(self.device)
                
                # Extract CLIP features
                img_features, txt_features = self.extract_clip_features(images, captions)
                
                # Forward pass through BCAN
                img_embeds, txt_embeds = self.model(img_features, txt_features)
                
                # Compute loss
                loss, loss_dict = self.criterion(img_embeds, txt_embeds)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
                self.optimizer.step()
                
                # Update stats
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'loss_lse': f'{loss_dict["loss_lse"]:.4f}',
                    'loss_sm': f'{loss_dict["loss_softmax"]:.4f}'
                })
                
                # Log step
                if (i + 1) % self.config.log_step == 0:
                    self.logger.log_step(epoch, i, loss_dict)
        
        avg_loss = total_loss / n_batches
        return avg_loss
    
    def validate(self, epoch):
        """Validate model"""
        print(f"\nValidating epoch {epoch}...")
        
        # Compute embeddings for all validation data
        img_embs, txt_embs = self.compute_embeddings(self.val_loader)
        
        # Evaluate
        results = self.evaluator.evaluate(img_embs, txt_embs)
        
        # Log results
        self.logger.log_validation(epoch, results)
        
        return results
    
    def compute_embeddings(self, data_loader):
        """Compute embeddings for all data in loader"""
        self.model.eval()
        
        img_embs_list = []
        txt_embs_list = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Computing embeddings'):
                images = batch['image'].to(self.device)
                captions = batch['caption'].to(self.device)
                
                # Extract CLIP features
                img_features, txt_features = self.extract_clip_features(images, captions)
                
                # Get BCAN embeddings
                img_embeds, txt_embeds = self.model(img_features, txt_features)
                
                img_embs_list.append(img_embeds.cpu())
                txt_embs_list.append(txt_embeds.cpu())
        
        # Concatenate all embeddings
        img_embs = torch.cat(img_embs_list, dim=0)
        txt_embs = torch.cat(txt_embs_list, dim=0)
        
        return img_embs, txt_embs
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_rsum': self.best_rsum,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.checkpoint_path,
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            torch.save(checkpoint, self.config.best_model_path)
            print(f"Saved best model with R@sum: {self.best_rsum:.2f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_rsum = checkpoint['best_rsum']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        for epoch in range(self.start_epoch, self.config.num_epochs):
            # Train epoch
            train_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch} - Average training loss: {train_loss:.4f}")
            
            # Validate
            val_results = self.validate(epoch)
            
            # Calculate R@sum
            rsum = val_results['i2t']['R@1'] + val_results['i2t']['R@5'] + val_results['i2t']['R@10'] + \
                   val_results['t2i']['R@1'] + val_results['t2i']['R@5'] + val_results['t2i']['R@10']
            
            print(f"Validation R@sum: {rsum:.2f}")
            
            # Save checkpoint
            is_best = rsum > self.best_rsum
            if is_best:
                self.best_rsum = rsum
            
            self.save_checkpoint(epoch, is_best)
        
        print("Training completed!")
        
        # Final test evaluation
        self.test()
    
    def test(self):
        """Evaluate on test set"""
        print("\nEvaluating on test set...")
        
        # Load best model
        if os.path.exists(self.config.best_model_path):
            self.load_checkpoint(self.config.best_model_path)
        
        # Compute test embeddings
        img_embs, txt_embs = self.compute_embeddings(self.test_loader)
        
        # Evaluate
        test_results = self.evaluator.evaluate(img_embs, txt_embs)
        
        # Print results
        print("\nTest Results:")
        print("Image-to-Text:")
        print(f"  R@1: {test_results['i2t']['R@1']:.2f}")
        print(f"  R@5: {test_results['i2t']['R@5']:.2f}")
        print(f"  R@10: {test_results['i2t']['R@10']:.2f}")
        print(f"  Mean R: {test_results['i2t']['mean_r']:.2f}")
        
        print("\nText-to-Image:")
        print(f"  R@1: {test_results['t2i']['R@1']:.2f}")
        print(f"  R@5: {test_results['t2i']['R@5']:.2f}")
        print(f"  R@10: {test_results['t2i']['R@10']:.2f}")
        print(f"  Mean R: {test_results['t2i']['mean_r']:.2f}")
        
        rsum = test_results['i2t']['R@1'] + test_results['i2t']['R@5'] + test_results['i2t']['R@10'] + \
               test_results['t2i']['R@1'] + test_results['t2i']['R@5'] + test_results['t2i']['R@10']
        
        print(f"\nR@sum: {rsum:.2f}")
        
        # Save test results
        results_path = os.path.join(self.config.log_path, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=4)


def main():
    # Load config
    sys.path.append('./configs')
    from config import Config
    config = Config()
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train model
    trainer.train()


if __name__ == '__main__':
    main()