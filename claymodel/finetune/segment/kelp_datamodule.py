"""
Modified DataModule for Kelp Classification dataset for segmentation tasks.

Key change: 80-20 train/validation split is now done at the DATA LEVEL, not folder level.
All data from non-test folders is collected, shuffled, then randomly split 80-20.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import lightning as L
import numpy as np
import torch
import rasterio as rio
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class KelpDataset(Dataset):
    """
    Dataset class for the Kelp segmentation dataset.
    [Dataset class remains unchanged]
    """

    def __init__(
        self, 
        data_paths: List[Tuple[str, str]], 
        normalization_params: Dict,
        target_size: int = 512,
        augment: bool = True,
    ):
        self.data_paths = data_paths
        self.target_size = target_size
        self.normalization_params = normalization_params
        
        # Create transforms
        self.augmentation_transform = self.create_augmentation_transform() if augment else None
        self.normalization_transform = self.create_normalization_transform()
        
        print(f"Dataset initialized with {len(self.data_paths)} samples")
        print(f"Target size: {target_size}x{target_size}")
        print(f"Augmentation: {'enabled' if augment else 'disabled'}")

    def create_augmentation_transform(self):
        """Create augmentation transforms using Albumentations."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(self.target_size, self.target_size, p=1.0)
        ])

    def create_normalization_transform(self):
        """Create normalization transform using provided normalization values."""
        mean = self.normalization_params['mean']
        std = self.normalization_params['std']
        
        return v2.Compose([
            v2.Normalize(mean=mean, std=std),
        ])

    def load_image(self, image_path: str) -> np.ndarray:
        """Load a 12-band Sentinel-2 TIFF image and extract some channels."""
        try:
            with rio.open(image_path) as src:
                image = src.read()  # Shape: (C, H, W)
                image = np.transpose(image, (1, 2, 0))  # Shape: (H, W, C)
                
                image = image[:, :, :10]  # Shape: (H, W, C)  The order of bands in my dataset: B2,B3,B4,B8,B5,B6,B7,B8A,B11,B12,Substrate,Bathymetry all resampled to 10m res. 
                                                #This should match waves in kelp_model.py.   [:,:,:10] means all S-2 bands
                return image.astype(np.float32)
                
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return np.zeros((self.target_size, self.target_size, 5), dtype=np.float32)

    def load_mask(self, mask_path: str) -> np.ndarray:
        """Load a binary mask."""
        try:
            with rio.open(mask_path) as src:
                mask = src.read(1)  # Read first band only
                mask = (mask > 0).astype(np.uint8)
                return mask
                
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            return np.zeros((self.target_size, self.target_size), dtype=np.uint8)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        image_path, mask_path = self.data_paths[idx]
        
        # Load image and mask
        image = self.load_image(image_path)  # (H, W, C)
        mask = self.load_mask(mask_path)     # (H, W)
        
        # Apply augmentations if enabled
        if self.augmentation_transform is not None:
            augmented = self.augmentation_transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Ensure consistent size even without augmentation
            if image.shape[:2] != (self.target_size, self.target_size):
                import cv2
                image = cv2.resize(image, (self.target_size, self.target_size))
                mask = cv2.resize(mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
        
        # Convert to PyTorch tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)
        mask = torch.from_numpy(mask).long()              # (H, W)
        
        # Apply normalization
        image = self.normalization_transform(image)
        
        sample = {
            "pixels": image,  # (C, 512, 512)
            "label": mask,    # (512, 512)
            "time": torch.zeros(4),    # Placeholder for time information
            "latlon": torch.zeros(4),  # Placeholder for latlon information  
        }
        
        return sample


class KelpDataModule(L.LightningDataModule):
    """
    DataModule class for the Kelp dataset with leave-one-folder-out cross-validation.
    
    MODIFIED: Train/validation split is now done at DATA LEVEL, not folder level.
    All data from non-test folders is collected and randomly split 80-20.
    """

    def __init__(
        self,
        data_root: str,
        normalization_params: Dict,
        test_folders: List[str],  # Required: which folders to hold out for testing
        batch_size: int = 4,
        num_workers: int = 4,
        target_size: int = 512,
        train_val_split: float = 0.8,
        random_seed: int = 42,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.normalization_params = normalization_params
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
        self.test_folders = test_folders
        self.train_val_split = train_val_split
        self.random_seed = random_seed
        
        # Set random seed for reproducible splits
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Initialize splits
        self.train_paths = None
        self.val_paths = None
        self.test_paths = None
        
        print(f"KelpDataModule initialized:")
        print(f"  Data root: {data_root}")
        print(f"  Test folders (held out): {test_folders}")
        print(f"  Batch size: {batch_size}")
        print(f"  Target size: {target_size}x{target_size}")
        print(f"  Train/Val split: {train_val_split:.1f}/{1-train_val_split:.1f}")
        print(f"  MODIFIED: Using DATA-LEVEL splitting (not folder-level)")
        print(f"  Normalization: mean={normalization_params['mean']}, std={normalization_params['std']}")

    def collect_data_paths(self, folder_names: List[str]) -> List[Tuple[str, str]]:
        """
        Collect all (image, mask) pairs from specified folders.
        [Method remains unchanged]
        """
        data_paths = []
        
        for folder_name in folder_names:
            folder_path = self.data_root / folder_name
            images_dir = folder_path / "images"
            masks_dir = folder_path / "masks"
            
            if not images_dir.exists() or not masks_dir.exists():
                print(f"Warning: Skipping {folder_name} - missing images or masks directory")
                continue
            
            # Get all image files
            image_files = list(images_dir.glob("*.tiff")) + list(images_dir.glob("*.tif"))
            
            for image_path in image_files:
                # Find corresponding mask
                image_name = image_path.stem
                # Try different mask naming patterns
                possible_mask_names = [
                    image_name.replace("_image", "_mask"),
                    image_name.replace("image", "mask"),
                    image_name + "_mask",
                    image_name,  # Same name different folder
                ]
                
                mask_path = None
                for mask_name in possible_mask_names:
                    potential_mask = masks_dir / f"{mask_name}.tiff"
                    if potential_mask.exists():
                        mask_path = potential_mask
                        break
                    potential_mask = masks_dir / f"{mask_name}.tif"
                    if potential_mask.exists():
                        mask_path = potential_mask
                        break
                
                if mask_path is not None:
                    data_paths.append((str(image_path), str(mask_path)))
                else:
                    print(f"Warning: No mask found for {image_path}")
        
        print(f"Collected {len(data_paths)} data pairs from {len(folder_names)} folders")
        return data_paths

    def create_train_val_split_from_data(self, all_data_paths: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        MODIFIED METHOD: Create train/val split at DATA LEVEL, not folder level.
        
        Args:
            all_data_paths: All (image, mask) pairs from non-test folders
            
        Returns:
            Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]: (train_paths, val_paths)
        """
        # Shuffle all data paths with fixed seed for reproducibility
        shuffled_paths = all_data_paths.copy()
        random.shuffle(shuffled_paths)
        
        # Split into train/val at the data level
        n_total = len(shuffled_paths)
        n_train = int(n_total * self.train_val_split)
        
        train_paths = shuffled_paths[:n_train]
        val_paths = shuffled_paths[n_train:]
        
        print(f"Data-level train/val split:")
        print(f"  Total samples from non-test folders: {n_total}")
        print(f"  Train samples: {len(train_paths)} ({len(train_paths)/n_total*100:.1f}%)")
        print(f"  Val samples: {len(val_paths)} ({len(val_paths)/n_total*100:.1f}%)")
        print(f"  Test folders (held out): {self.test_folders}")
        
        return train_paths, val_paths

    def setup(self, stage=None):
        """
        MODIFIED: Setup datasets using data-level train/val split.
        """
        if stage in {"fit", None}:
            # Get all available folders except the test folder
            all_folders = [f.name for f in self.data_root.iterdir() if f.is_dir()]
            all_folders.sort()  # For reproducibility
            
            # Remove the test folders from the list
            train_val_folders = [f for f in all_folders if f not in self.test_folders]
            
            print(f"Non-test folders used for train/val: {train_val_folders}")
            
            # Collect ALL data from the non-test folders
            all_train_val_data = self.collect_data_paths(train_val_folders)
            
            # Split at DATA level, not folder level
            self.train_paths, self.val_paths = self.create_train_val_split_from_data(all_train_val_data)
            
            # Create datasets
            self.trn_ds = KelpDataset(
                self.train_paths,
                self.normalization_params,
                target_size=self.target_size,
                augment=True,  # Augmentation for training
            )
            
            self.val_ds = KelpDataset(
                self.val_paths,
                self.normalization_params,
                target_size=self.target_size,
                augment=False,  # No augmentation for validation
            )
            
        if stage in {"test", None}:
            # Test on the held-out folder
            self.test_paths = self.collect_data_paths([self.test_folders])
            self.test_ds = KelpDataset(
                self.test_paths,
                self.normalization_params,
                target_size=self.target_size,
                augment=False,  # No augmentation for testing
            )

    def train_dataloader(self):
        """Create DataLoader for training data."""
        return DataLoader(
            self.trn_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        """Create DataLoader for validation data."""
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        """Create DataLoader for test data."""
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )