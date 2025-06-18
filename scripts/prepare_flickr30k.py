#!/usr/bin/env python3
"""
Script to prepare Flickr dataset splits (supports both Flickr8k and Flickr30k)
"""

import os
import random
import argparse
from collections import defaultdict

def create_splits(data_path, dataset_name='flickr30k', train_ratio=0.9, val_ratio=0.05, test_ratio=0.05):
    """
    Create train/val/test splits for Flickr dataset
    """
    # Determine image directory based on dataset
    if 'flickr30k' in dataset_name.lower():
        images_dir = os.path.join(data_path, 'flickr30k-images')
    else:  # flickr8k
        images_dir = os.path.join(data_path, 'Flicker8k_Dataset')
    
    # Read all image files
    all_images = [f.replace('.jpg', '') for f in os.listdir(images_dir) if f.endswith('.jpg')]
    
    # Shuffle and split
    random.seed(42)  # For reproducibility
    random.shuffle(all_images)
    
    n_total = len(all_images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_images = all_images[:n_train]
    val_images = all_images[n_train:n_train+n_val]
    test_images = all_images[n_train+n_val:]
    
    print(f"Total images: {n_total}")
    print(f"Train: {len(train_images)}")
    print(f"Val: {len(val_images)}")
    print(f"Test: {len(test_images)}")
    
    # Save splits
    splits = {
        'train_images.txt': train_images,
        'val_images.txt': val_images,
        'test_images.txt': test_images
    }
    
    for filename, image_list in splits.items():
        filepath = os.path.join(data_path, filename)
        with open(filepath, 'w') as f:
            for img in image_list:
                f.write(f"{img}\n")
        print(f"Saved {filepath}")

def verify_dataset(data_path, dataset_name='flickr30k'):
    """
    Verify dataset structure and files
    """
    if 'flickr30k' in dataset_name.lower():
        required_files = [
            'results_20130124.token',
            'flickr30k-images'
        ]
        captions_file = 'results_20130124.token'
        images_dir = 'flickr30k-images'
    else:  # flickr8k
        required_files = [
            'Flickr8k.token.txt',
            'Flicker8k_Dataset'
        ]
        captions_file = 'Flickr8k.token.txt'
        images_dir = 'Flicker8k_Dataset'
    
    missing = []
    for item in required_files:
        path = os.path.join(data_path, item)
        if not os.path.exists(path):
            missing.append(item)
    
    if missing:
        print(f"Error: Missing required files/directories: {missing}")
        return False
    
    # Check captions file
    captions_path = os.path.join(data_path, captions_file)
    caption_count = 0
    image_caption_map = defaultdict(list)
    
    with open(captions_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                img_id = parts[0].split('#')[0]
                image_caption_map[img_id].append(parts[1])
                caption_count += 1
    
    print(f"Found {len(image_caption_map)} images with {caption_count} total captions")
    print(f"Average captions per image: {caption_count / len(image_caption_map):.2f}")
    
    # Check images
    images_path = os.path.join(data_path, images_dir)
    image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]
    print(f"Found {len(image_files)} image files")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Prepare Flickr dataset')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to Flickr dataset')
    parser.add_argument('--dataset_name', type=str, default='flickr30k',
                        choices=['flickr8k', 'flickr30k'],
                        help='Dataset name (flickr8k or flickr30k)')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.05,
                        help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.05,
                        help='Test set ratio')
    
    args = parser.parse_args()
    
    # Verify dataset
    print(f"Verifying {args.dataset_name} dataset...")
    if not verify_dataset(args.data_path, args.dataset_name):
        return
    
    # Create splits
    print(f"\nCreating splits for {args.dataset_name}...")
    create_splits(
        args.data_path,
        args.dataset_name,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )
    
    print(f"\n{args.dataset_name} dataset preparation complete!")

if __name__ == '__main__':
    main()