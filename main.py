#!/usr/bin/env python3
"""
Main entry point for CLIP-BCAN training and evaluation
"""

import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'configs'))

from config import Config
from src.train import Trainer

def main():
    parser = argparse.ArgumentParser(description='CLIP-BCAN Cross-Modal Retrieval')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='/path/to/flickr8k',
                        help='Path to Flickr dataset')
    parser.add_argument('--dataset_name', type=str, default='flickr8k',
                        choices=['flickr8k', 'flickr30k'],
                        help='Dataset name (flickr8k or flickr30k)')
    
    # Model arguments
    parser.add_argument('--clip_model', type=str, default='ViT-B/16',
                        help='CLIP model variant')
    parser.add_argument('--embed_size', type=int, default=1024,
                        help='BCAN embedding size')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.2,
                        help='Contrastive loss margin')
    
    # Other arguments
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--test_only', action='store_true',
                        help='Only run test evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    
    # Override config with command line arguments
    config.data_path = args.data_path
    config.dataset_name = args.dataset_name
    config.clip_model = args.clip_model
    config.embed_size = args.embed_size
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.learning_rate = args.learning_rate
    config.margin = args.margin
    config.device = args.device
    
    # Update derived paths based on dataset
    if 'flickr30k' in config.dataset_name.lower():
        config.images_path = os.path.join(config.data_path, 'flickr30k-images')
        config.captions_path = os.path.join(config.data_path, 'results_20130124.token')
    else:  # flickr8k
        config.images_path = os.path.join(config.data_path, 'Flicker8k_Dataset')
        config.captions_path = os.path.join(config.data_path, 'Flickr8k.token.txt')
    
    print("Configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Run training or testing
    if args.test_only:
        trainer.test()
    else:
        trainer.train()

if __name__ == '__main__':
    main()