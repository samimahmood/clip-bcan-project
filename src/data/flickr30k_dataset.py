import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import clip
import numpy as np

class FlickrDataset(Dataset):
    def __init__(self, data_path, split='train', clip_model=None, preprocess=None, dataset_name='flickr30k'):
        """
        Args:
            data_path: Path to Flickr dataset
            split: 'train', 'val', or 'test'
            clip_model: CLIP model for text encoding
            preprocess: CLIP preprocessing function for images
            dataset_name: 'flickr8k' or 'flickr30k'
        """
        self.data_path = data_path
        self.split = split
        self.clip_model = clip_model
        self.preprocess = preprocess
        self.dataset_name = dataset_name.lower()
        
        # Set paths based on dataset
        if 'flickr30k' in self.dataset_name:
            self.images_dir = os.path.join(data_path, 'flickr30k-images')
            self.captions_file = os.path.join(data_path, 'results_20130124.token')
        else:  # flickr8k
            self.images_dir = os.path.join(data_path, 'Flicker8k_Dataset')
            self.captions_file = os.path.join(data_path, 'Flickr8k.token.txt')
        
        # Load annotations
        self.images, self.captions = self._load_annotations()
        
        # Create caption to image index mapping
        self.caption_to_image = []
        for i, img_captions in enumerate(self.captions):
            self.caption_to_image.extend([i] * len(img_captions))
    
    def _load_annotations(self):
        """Load Flickr annotations based on split"""
        # Define split files (you'll need to create these based on standard splits)
        split_files = {
            'train': 'train_images.txt',
            'val': 'val_images.txt', 
            'test': 'test_images.txt'
        }
        
        # For Flickr8k, check if standard split files exist
        if 'flickr8k' in self.dataset_name:
            flickr8k_splits = {
                'train': 'Flickr_8k.trainImages.txt',
                'val': 'Flickr_8k.devImages.txt',
                'test': 'Flickr_8k.testImages.txt'
            }
            # Check if Flickr8k split files exist
            if os.path.exists(os.path.join(self.data_path, flickr8k_splits['train'])):
                split_files = flickr8k_splits
        
        # Read image IDs for this split
        split_file = os.path.join(self.data_path, split_files[self.split])
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                image_ids = [line.strip() for line in f]
                # Remove .jpg extension if present
                image_ids = [img.replace('.jpg', '') for img in image_ids]
        else:
            # If split files don't exist, use all images and split manually
            print(f"Split file {split_file} not found. Using manual split.")
            image_ids = self._create_manual_split()
        
        # Load captions
        image_captions = {}
        
        with open(self.captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    # Handle both formats: with and without .jpg
                    img_id = parts[0].split('#')[0]
                    img_id = img_id.replace('.jpg', '')  # Remove .jpg if present
                    caption = parts[1]
                    
                    if img_id not in image_captions:
                        image_captions[img_id] = []
                    image_captions[img_id].append(caption)
        
        # Debug: print some info
        print(f"Total captions loaded: {len(image_captions)}")
        print(f"Sample image IDs from captions: {list(image_captions.keys())[:5]}")
        print(f"Sample image IDs from split: {image_ids[:5]}")
        
        # Filter based on split
        images = []
        captions = []
        
        for img_id in image_ids:
            if img_id in image_captions:
                images.append(img_id)
                captions.append(image_captions[img_id][:5])  # Use first 5 captions
            else:
                # Debug missing images
                if len(images) < 5:  # Only print first few
                    print(f"Warning: Image {img_id} not found in captions")
        
        print(f"Loaded {len(images)} images with {sum(len(c) for c in captions)} captions for {self.split}")
        
        return images, captions
    
    def _create_manual_split(self):
        """Create manual train/val/test split if split files don't exist"""
        # Get all image files
        all_images = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]
        all_images.sort()
        
        # Remove .jpg extension
        all_images = [img.replace('.jpg', '') for img in all_images]
        
        # Different split ratios for different datasets
        n_total = len(all_images)
        
        if 'flickr8k' in self.dataset_name:
            # Flickr8k standard split: 6k train, 1k val, 1k test
            n_test = 1000
            n_val = 1000
        else:  # flickr30k
            # Flickr30k standard split: 29k train, 1k val, 1k test
            n_test = 1000
            n_val = 1000
        
        n_train = n_total - n_test - n_val
        
        if self.split == 'train':
            return all_images[:n_train]
        elif self.split == 'val':
            return all_images[n_train:n_train+n_val]
        else:  # test
            return all_images[n_train+n_val:]
    
    def __len__(self):
        return len(self.caption_to_image)
    
    def __getitem__(self, index):
        # Get image index from caption index
        img_idx = self.caption_to_image[index]
        
        # Get caption index within the image's captions
        caption_idx = index - sum(len(self.captions[i]) for i in range(img_idx))
        
        # Load image
        img_path = os.path.join(self.images_dir, self.images[img_idx] + '.jpg')
        image = Image.open(img_path).convert('RGB')
        
        if self.preprocess:
            image = self.preprocess(image)
        
        # Get caption
        caption = self.captions[img_idx][caption_idx]
        
        # Tokenize caption
        if self.clip_model:
            caption_tokens = clip.tokenize([caption], truncate=True)[0]
        else:
            caption_tokens = caption
        
        return {
            'image': image,
            'caption': caption_tokens,
            'caption_text': caption,
            'image_id': img_idx,
            'caption_id': index
        }
    
    def get_all_captions(self):
        """Get all captions for evaluation"""
        all_captions = []
        for captions_list in self.captions:
            all_captions.extend(captions_list)
        return all_captions


class FlickrDataLoader:
    def __init__(self, config):
        self.config = config
        
        # Load CLIP model and preprocessing
        self.clip_model, self.preprocess = clip.load(config.clip_model, device=config.device)
        self.clip_model.eval()
        
    def get_loaders(self):
        """Get train, val, and test data loaders"""
        # Create datasets
        train_dataset = FlickrDataset(
            self.config.data_path,
            split='train',
            clip_model=self.clip_model,
            preprocess=self.preprocess,
            dataset_name=self.config.dataset_name
        )
        
        val_dataset = FlickrDataset(
            self.config.data_path,
            split='val',
            clip_model=self.clip_model,
            preprocess=self.preprocess,
            dataset_name=self.config.dataset_name
        )
        
        test_dataset = FlickrDataset(
            self.config.data_path,
            split='test',
            clip_model=self.clip_model,
            preprocess=self.preprocess,
            dataset_name=self.config.dataset_name
        )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader, self.clip_model