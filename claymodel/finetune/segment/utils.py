import rasterio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as trns
import torch.nn as nn

from torchmetrics import JaccardIndex
from sklearn.model_selection import train_test_split

import os
import albumentations as A

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
import pytorch_lightning as pl

# to save the results

import tifffile as tiff  # Use tifffile to handle TIFF files
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

def normalize_input_mean_std(image_hwc, mean_per_channel, std_per_channel, epsilon=1e-8):
    """Applies mean and std normalization to the input image (H, W, C) at once."""
    image_hwc = np.nan_to_num(image_hwc).astype(np.float32)  # Handle NaNs and ensure float type
    mean = np.array(mean_per_channel, dtype=np.float32)[np.newaxis, np.newaxis, :]
    std = np.array(std_per_channel, dtype=np.float32)[np.newaxis, np.newaxis, :]
    normalized_image = (image_hwc - mean) / (std + epsilon)
    return normalized_image

class SatelliteDataset(BaseDataset):
    CLASSES = ["water", "kelp", "land"]

    def __init__(self, image_paths, mask_paths, classes=None, augmentation=None, mean=None, std=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augmentation = augmentation
        self.mean = mean
        self.std = std
        self.calculated_mean = None
        self.calculated_std = None

        if classes is None:
            classes = self.CLASSES
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.index_calculators = {
            "ndvi": self.calculate_ndvi,
            "ndwi": self.calculate_ndwi,
            "gndvi": self.calculate_gndvi,
            "clgreen": self.calculate_chlorophyll_index_green,
            "ndvire": self.calculate_ndvi_re, #Normalized Difference of Red edgeand Blue
            #"ndrb": self.calculate_ndrb, #Normalized Difference of Red and Blue
            #"mgvi": self.calculate_mgvi, #Modified Green Red Vegetation Index (MGVI)
            #"mpri": self.calculate_mpri, #Modified Photochemical Reflectance Index (MPRI)
            #"rgbvi": self.calculate_rgbvi, #Red Green Blue Vegetation Index (RGBVI)
            #"gli": self.calculate_gli, #Green Leaf Index (GLI)
            #"gi": self.calculate_gi, #Greenness Index (GI)
            #"br": self.calculate_blue_red, #Blue/Red
            #"exg": self.calculate_exg, #Excess of Green (ExG)
            #"vari": self.calculate_vari, #Visible Atmospherically Resistant Index (VARI)
            #"tvi": self.calculate_tvi, #Triangular Vegetation Index (TVI)
            #"rdvi": self.calculate_rdvi, #Renormalized Difference Vegetation Index (RDVI)
            #"ndreb": self.calculate_ndreb, #Normalized Difference Red-edge Blue (NDREB)
            #"evi": self.calculate_evi, #Enhanced Vegetation Index (EVI)
            #"cig": self.calculate_cig,  #Green Chlorophyll Index (CIG)
            #"blue_rededge": self.calculate_blue_rededge, #Blue/Red-edge
            #"bnir": self.calculate_blue_nir, #Blue/NIR
            #"rb": self.calculate_red_minus_blue, #R-B
            #"bndvi": self.calculate_bndvi, #Blue NDVI
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # --- 1. Read Image ---
        img_path = self.image_paths[index]
        with rasterio.open(img_path) as src_img:
            image = src_img.read([1, 2, 3, 4,5,6,7,8,9,10,11,12]) # -> (C, H, W) = (6, H, W) 1-4:10m bands, 5-10: 20m bands resampled to 10m, 11:substrate 12: bathymetry, 13: S1 VV channel, and 14: S1 VH channel

            # --- FIXED PREPROCESSING ---
            # # Modify bathymetry (channel index 5 in CHW format)
            # bathy_mask_gt = image[6, :, :] > 10
            # bathy_mask_lt = image[6, :, :] < -100
            # image[6, :, :][bathy_mask_gt | bathy_mask_lt] = -2000 # Combine conditions

            # # Modify substrate (channel index 4 in CHW format)
            # subs_mask = image[5, :, :] != 1
            # image[5, :, :][subs_mask] = 0
            # # --- END FIX ---

            image_hwc = np.transpose(image, (1, 2, 0)).astype(np.float32) # -> (H, W, 6)

        # --- 2. Calculate Indices ---
        indices_list = []
        for _, calculator in self.index_calculators.items():
            idx = calculator(image_hwc)
            indices_list.append(idx[..., np.newaxis])

        image_with_indices = np.concatenate([image_hwc] + indices_list, axis=-1)

        # --- 3. Read Mask ---
        mask_path = self.mask_paths[index]
        mask = self.read_and_process_mask(mask_path) # -> (H, W, num_classes)

        # --- 4. Apply Normalization ---
        if self.mean is not None and self.std is not None:
            image_with_indices = normalize_input_mean_std(image_with_indices, mean_per_channel=self.mean, std_per_channel=self.std)

        # --- 5. Apply Augmentation ---
        if self.augmentation:  # Proceeds to call whatever is stored in self.augmentation
            # Python expects self.augmentation to be a callable object (something that can be called using parentheses ()), which a function is. 
            # The image and mask variables are passed as arguments to this callable. In the case of albumentation, self.augmentation holds the
            # A.Compose object. When you call it with (image=image_with_indices, mask=mask), the A.Compose object's __call__ method  (which is what makes an
            # object callable) is executed. This method internally applies the defined horizontal and vertical flips (with their respective probabilities)
            # to the provided image and mask and returns a dictionary like {'image': augmented_image, 'mask': augmented_mask}.
            sample = self.augmentation(image=image_with_indices, mask=mask) 
            image_with_indices = sample['image']
            mask = sample['mask']

        # --- 6. Final Transpose ---
        image_final = np.transpose(image_with_indices, (2, 0, 1))
        mask_final = np.transpose(mask, (2, 0, 1))

        return image_final.astype(np.float32), mask_final.astype(np.float32)


    def read_and_process_mask(self, mask_path):
        with rasterio.open(mask_path) as src_mask:
            mask = src_mask.read(1).astype(int) # -> (H, W)
            masks = [(mask == v) for v in self.class_values]
            return np.stack(masks, axis=-1).astype("float") #-> (H, W, num_classes)

    # --- Index Calculation Methods ---
    def calculate_ndvi(self, image_hwc):
        nir = image_hwc[..., 3]
        red = image_hwc[..., 2]
        return (nir - red) / (nir + red + 1e-10)

    def calculate_ndwi(self, image_hwc):
        green = image_hwc[..., 1]
        nir = image_hwc[..., 3]
        return (green - nir) / (green + nir + 1e-10)

    def calculate_gndvi(self, image_hwc):
        nir = image_hwc[..., 3]
        green = image_hwc[..., 1]
        return (nir - green) / (nir + green + 1e-10)

    def calculate_chlorophyll_index_green(self, image_hwc):
        nir = image_hwc[..., 3]
        green = image_hwc[..., 1]
        return (nir / (green + 1e-10)) - 1

    def calculate_ndvi_re(self, image_hwc):
        re = image_hwc[..., 4]
        red = image_hwc[..., 2]
        return (re - red) / (re + red + 1e-10)

    def calculate_evi(self, image_hwc):
        nir = image_hwc[..., 3]
        red = image_hwc[..., 2]
        blue = image_hwc[..., 0]
        return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-10)

    def calculate_sr(self, image_hwc):
        nir = image_hwc[..., 3]
        red = image_hwc[..., 2]
        return nir / (red + 1e-10)

    def calculate_ndrb(self, image_hwc):
        return (image_hwc[..., 2] - image_hwc[..., 0]) / (image_hwc[..., 2] + image_hwc[..., 0] + 1e-10)

    def calculate_mgvi(self, image_hwc):
        return (image_hwc[..., 1]**2 - image_hwc[..., 2]**2) / (image_hwc[..., 1]**2 + image_hwc[..., 2]**2 + 1e-10)

    def calculate_mpri(self, image_hwc):
        return (image_hwc[..., 1] - image_hwc[..., 2]) / (image_hwc[..., 1] + image_hwc[..., 2] + 1e-10)

    def calculate_rgbvi(self, image_hwc):
        return (image_hwc[..., 1] - image_hwc[..., 0] * image_hwc[..., 2]) / (image_hwc[..., 1]**2 + image_hwc[..., 0] * image_hwc[..., 2] + 1e-10)

    def calculate_gli(self, image_hwc):
        return (2 * image_hwc[..., 1] - image_hwc[..., 2] - image_hwc[..., 0]) / (2 * image_hwc[..., 1] + image_hwc[..., 2] + image_hwc[..., 0] + 1e-10)

    def calculate_gi(self, image_hwc):
        return image_hwc[..., 1] / (image_hwc[..., 2] + 1e-10)

    def calculate_blue_red(self, image_hwc):
        return image_hwc[..., 0] / (image_hwc[..., 2] + 1e-10)

    def calculate_red_minus_blue(self, image_hwc):
        return image_hwc[..., 2] - image_hwc[..., 0]

    def calculate_exg(self, image_hwc):
        return 2 * image_hwc[..., 1] - image_hwc[..., 2] - image_hwc[..., 0]

    def calculate_vari(self, image_hwc):
        return (image_hwc[..., 1] - image_hwc[..., 2]) / (image_hwc[..., 1] + image_hwc[..., 2] - image_hwc[..., 0] + 1e-10)

    def calculate_tvi(self, image_hwc):
        return (120 * (image_hwc[..., 4] - image_hwc[..., 1]) - 200 * (image_hwc[..., 2] - image_hwc[..., 1])) / 2

    def calculate_rdvi(self, image_hwc):
        return (image_hwc[..., 3] - image_hwc[..., 2]) / np.sqrt(image_hwc[..., 3] + image_hwc[..., 2] + 1e-10)

    def calculate_ndreb(self, image_hwc):
        return (image_hwc[..., 4] - image_hwc[..., 0]) / (image_hwc[..., 4] + image_hwc[..., 0] + 1e-10)

    def calculate_cig(self, image_hwc):
        return (image_hwc[..., 3] / (image_hwc[..., 1] + 1e-10)) - 1

    def calculate_blue_rededge(self, image_hwc):
        return image_hwc[..., 0] / (image_hwc[..., 4] + 1e-10)

    def calculate_blue_nir(self, image_hwc):
        return image_hwc[..., 0] / (image_hwc[..., 3] + 1e-10)

    def calculate_bndvi(self, image_hwc):
        nir = image_hwc[..., 3]
        blue = image_hwc[..., 0]
        return (nir - blue) / (nir + blue + 1e-10)