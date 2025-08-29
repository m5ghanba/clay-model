#!/usr/bin/env python3
"""
Debug script to inspect KelpDataModule output
"""
import torch
from claymodel.finetune.segment.kelp_datamodule import KelpDataModule

def debug_kelp_data():
    """Debug the data format from KelpDataModule"""
    
    # Initialize your datamodule with num_workers=0 for Windows
    dm = KelpDataModule(
        data_root='C:/Annotated Dataset/L2A Data/Tiles10and20mbandsSubsBathy_G0Percent_stride256',
        test_folder='20230816T191911_20230816T192348_T10UCA',
        normalization_params={
            'mean': [192.710564, 245.262772, 132.882966, 915.828301, 298.188553], 
            'std': [151.37001, 208.243277, 194.224862, 1242.17269, 368.341824]
        },
        batch_size=1,
        target_size=512,
        num_workers=0  # Important for Windows
    )
    
    dm.setup('fit')
    train_loader = dm.train_dataloader()
    
    print("Getting first batch...")
    batch = next(iter(train_loader))
    
    print('\n=== Batch Analysis ===')
    print('Batch keys:', list(batch.keys()))
    
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f'{k}: shape={v.shape}, dtype={v.dtype}, device={v.device}')
            if k == 'pixels':  # The main image data
                print(f'  {k} min/max: {v.min():.3f}/{v.max():.3f}')
            if v.numel() < 20:  # Small tensors, show values
                print(f'  {k} values: {v}')
        else:
            print(f'{k}: {type(v)}, value={v}')
    
    # Check if wavelengths exist
    if 'wavelengths' in batch:
        waves = batch['wavelengths']
        print(f'\nWavelengths found: {waves}')
    else:
        print('\nWARNING: No wavelengths found in batch!')
        print('Available keys:', list(batch.keys()))
    
    return batch

if __name__ == '__main__':
    batch = debug_kelp_data()