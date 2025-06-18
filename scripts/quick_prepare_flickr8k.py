#!/usr/bin/env python3
"""
Quick script to prepare Flickr8k dataset splits
"""

import os
import random

def prepare_flickr8k(data_path):
    """Create train/val/test splits for Flickr8k"""
    
    # Check if standard Flickr8k split files exist
    standard_splits = {
        'train': 'Flickr_8k.trainImages.txt',
        'val': 'Flickr_8k.devImages.txt',
        'test': 'Flickr_8k.testImages.txt'
    }
    
    # Check if files already exist
    all_exist = True
    for split, filename in standard_splits.items():
        filepath = os.path.join(data_path, filename)
        if not os.path.exists(filepath):
            print(f"Standard split file {filename} not found")
            all_exist = False
        else:
            print(f"Found {filename}")
    
    if all_exist:
        print("\nAll standard split files exist!")
        return
    
    # If not, create manual splits
    print("\nCreating manual splits...")
    
    # Get all images
    images_dir = os.path.join(data_path, 'Flicker8k_Dataset')
    if not os.path.exists(images_dir):
        print(f"ERROR: Images directory not found at {images_dir}")
        return
    
    all_images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    print(f"Found {len(all_images)} images")
    
    # Shuffle for random split
    random.seed(42)
    random.shuffle(all_images)
    
    # Standard Flickr8k split: 6000 train, 1000 val, 1000 test
    n_train = 6000
    n_val = 1000
    n_test = 1000
    
    train_images = all_images[:n_train]
    val_images = all_images[n_train:n_train+n_val]
    test_images = all_images[n_train+n_val:n_train+n_val+n_test]
    
    # Save split files
    splits = {
        'train_images.txt': train_images,
        'val_images.txt': val_images,
        'test_images.txt': test_images
    }
    
    for filename, image_list in splits.items():
        filepath = os.path.join(data_path, filename)
        with open(filepath, 'w') as f:
            for img in image_list:
                # Remove .jpg extension
                img_id = img.replace('.jpg', '')
                f.write(f"{img_id}\n")
        print(f"Created {filename} with {len(image_list)} images")
    
    print("\nDataset preparation complete!")
    
    # Also check captions file
    captions_file = os.path.join(data_path, 'Flickr8k.token.txt')
    if os.path.exists(captions_file):
        with open(captions_file, 'r') as f:
            n_lines = sum(1 for _ in f)
        print(f"\nFound captions file with {n_lines} lines")
    else:
        print(f"\nWARNING: Captions file not found at {captions_file}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "/Users/samimahmood/Documents/Work/Personal/clip-bcan-project/Flicker8k"
    
    prepare_flickr8k(data_path)