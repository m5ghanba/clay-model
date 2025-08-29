"""
Clay Segmentor for semantic segmentation tasks.

Attribution:
Decoder from Segformer: Simple and Efficient Design for Semantic Segmentation
with Transformers
Paper URL: https://arxiv.org/abs/2105.15203

Modified for Kelp Classification with 512x512 input support.

"""

import re
import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from claymodel.model import Encoder


class SegmentEncoder(Encoder):
    """
    Encoder class for segmentation tasks, incorporating a feature pyramid
    network (FPN) with support for variable input sizes through positional
    embedding interpolation.

    Attributes:
        feature_maps (list): Indices of layers to be used for generating
        feature maps.
        ckpt_path (str): Path to the clay checkpoint file.
        pretrained_size (int): Original training size for positional embeddings (224).
        current_size (int): Current input size (e.g., 512).
    """

    def __init__(  # noqa: PLR0913
        self,
        mask_ratio,
        patch_size,
        shuffle,
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
        ckpt_path=None,
        pretrained_size=224,
        current_size=512,
    ):
        super().__init__(
            mask_ratio,
            patch_size,
            shuffle,
            dim,
            depth,
            heads,
            dim_head,
            mlp_ratio,
        )

        # Store size information for positional embedding interpolation
        self.pretrained_size = pretrained_size
        self.current_size = current_size
        self.pretrained_patches = pretrained_size // patch_size  # 224//8 = 28
        self.current_patches = current_size // patch_size  # 512//8 = 64

        # Set device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # Load model from checkpoint if provided
        self.load_from_ckpt(ckpt_path)

    def interpolate_pos_embed(self, pos_embed_checkpoint, new_size):
        """
        Interpolate positional embeddings for different input sizes.
        
        Args:
            pos_embed_checkpoint (torch.Tensor): Original positional embeddings 
                from checkpoint with shape [1, N+1, D] where N = 28*28 = 784
            new_size (int): New spatial size (e.g., 64 for 64x64 patches)
            
        Returns:
            torch.Tensor: Interpolated positional embeddings [1, new_N+1, D]
        """
        # Original: [1, 785, 1024] -> [1, 784, 1024] (remove cls token) + [1, 1, 1024] (cls token)
        embedding_size = pos_embed_checkpoint.shape[-1]  # 1024
        num_extra_tokens = 1  # cls token
        
        # Extract class token and spatial tokens
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]  # [1, 1, 1024]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]    # [1, 784, 1024]
        
        # Get original spatial dimensions
        orig_size = int(math.sqrt(pos_tokens.shape[1]))  # 28
        
        if orig_size != new_size:
            print(f"Interpolating positional embeddings from {orig_size}x{orig_size} to {new_size}x{new_size}")
            
            # Reshape to spatial dimensions: [1, 784, 1024] -> [1, 1024, 28, 28]
            pos_tokens = pos_tokens.reshape(1, orig_size, orig_size, embedding_size)
            pos_tokens = pos_tokens.permute(0, 3, 1, 2)  # [1, 1024, 28, 28]
            
            # Interpolate to new size: [1, 1024, 28, 28] -> [1, 1024, 64, 64]
            pos_tokens = F.interpolate(
                pos_tokens, 
                size=(new_size, new_size), 
                mode='bicubic', 
                align_corners=False
            )
            
            # Reshape back: [1, 1024, 64, 64] -> [1, 4096, 1024]
            pos_tokens = pos_tokens.permute(0, 2, 3, 1)  # [1, 64, 64, 1024]
            pos_tokens = pos_tokens.reshape(1, new_size * new_size, embedding_size)
        
        # Concatenate class token and spatial tokens
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        return new_pos_embed

    def load_from_ckpt(self, ckpt_path):
        """
        Load the model's state from a checkpoint file with positional embedding interpolation.

        Args:
            ckpt_path (str): The path to the checkpoint file.
        """
        if ckpt_path:
            # Load checkpoint
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state_dict = ckpt.get("state_dict")

            # Prepare new state dict with the desired subset and naming
            new_state_dict = {}
            for name, param in state_dict.items():
                if name.startswith("model.encoder"):
                    new_name = re.sub(r"^model\.encoder\.", "", name)
                    
                    # Handle positional embeddings specially
                    if new_name == "pos_embed" and self.current_size != self.pretrained_size:
                        # Interpolate positional embeddings
                        interpolated_pos_embed = self.interpolate_pos_embed(
                            param, self.current_patches
                        )
                        new_state_dict[new_name] = interpolated_pos_embed
                        print(f"Interpolated pos_embed from {param.shape} to {interpolated_pos_embed.shape}")
                    else:
                        new_state_dict[new_name] = param

            # Load the modified state dict into the model
            model_state_dict = self.state_dict()
            for name, param in new_state_dict.items():
                if (
                    name in model_state_dict
                    and param.size() == model_state_dict[name].size()
                ):
                    model_state_dict[name].copy_(param)
                    print(f"Loaded parameter: {name} with size {param.size()}")
                else:
                    print(f"No matching parameter for {name} with size {param.size()}")

            # Freeze the loaded parameters
            for name, param in self.named_parameters():
                if name in new_state_dict:
                    param.requires_grad = False
                    
        print(f"Encoder configured for {self.current_size}x{self.current_size} input "
              f"({self.current_patches}x{self.current_patches} patches)")

    def forward(self, datacube):
        """
        Forward pass of the SegmentEncoder.

        Args:
            datacube (dict): A dictionary containing the input datacube and
                meta information like time, latlon, gsd & wavelenths.

        Returns:
            list: A list of feature maps extracted from the datacube.
        """
        cube, time, latlon, gsd, waves = (
            datacube["pixels"],  # [B C H W]
            datacube["time"],  # [B 2]
            datacube["latlon"],  # [B 2]
            datacube["gsd"],  # 1
            datacube["waves"],  # [N]
        )

        B, C, H, W = cube.shape


        # print(f"After unbatching - gsd: {gsd.shape if hasattr(gsd, 'shape') else 'scalar'}, waves: {waves.shape}")


        # Verify input size matches expected size
        if H != self.current_size or W != self.current_size:
            print(f"Warning: Input size {H}x{W} doesn't match expected {self.current_size}x{self.current_size}")

        # Patchify and create embeddings per patch
        patches, waves_encoded = self.to_patch_embed(cube, waves)  # [B L D]
        # print(f"patches: {patches.shape}, waves_encoded: {waves_encoded.shape}")
        # print(f"self.dim: {self.dim}")
        patches = self.add_encodings(patches, time, latlon, gsd)  # [B L D]

        # Add class tokens
        cls_tokens = repeat(self.cls_token, "1 1 D -> B 1 D", B=B)  # [B 1 D]
        patches = torch.cat((cls_tokens, patches), dim=1)  # [B (1 + L) D]

        patches = self.transformer(patches)
        patches = patches[:, 1:, :]  # [B L D] - Remove cls token
        # print(f"Final patches: {patches.shape}")

        return patches


class Segmentor(nn.Module):
    """
    Clay Segmentor class that combines the Encoder with decoder layers for semantic
    segmentation, with support for variable input sizes.

    Attributes:
        num_classes (int): Number of output classes for segmentation.
        ckpt_path (str): Path to the checkpoint file.
        input_size (int): Input image size (default: 512).
    """

    def __init__(self, num_classes, ckpt_path, input_size=512):
        super().__init__()
        
        self.input_size = input_size
        
        # Default values are for the clay mae base model.
        self.encoder = SegmentEncoder(
            mask_ratio=0.0,
            patch_size=8,
            shuffle=False,
            dim=1024,
            depth=24,
            heads=16,
            dim_head=64,
            mlp_ratio=4.0,
            ckpt_path=ckpt_path,
            pretrained_size=224,  # Original Clay training size
            current_size=input_size,  # Your desired input size
        )

        # Freeze the encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Define layers after the encoder
        D = self.encoder.dim  # embedding dimension (1024)
        hidden_dim = 512
        C_out = 64
        r = self.encoder.patch_size  # upscale factor (patch_size = 8)

        # Decoder layers
        self.conv1 = nn.Conv2d(D, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv_ps = nn.Conv2d(hidden_dim, C_out * r * r, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=r)
        self.conv_out = nn.Conv2d(C_out, num_classes, kernel_size=3, padding=1)

        print(f"Segmentor initialized for {input_size}x{input_size} input, {num_classes} classes")

    def forward(self, datacube):
        """
        Forward pass of the Segmentor.

        Args:
            datacube (dict): A dictionary containing the input datacube and
                meta information like time, latlon, gsd & wavelenths.

        Returns:
            torch.Tensor: The segmentation logits.
        """
        cube = datacube["pixels"]  # [B C H_in W_in]
        B, C, H_in, W_in = cube.shape

        # Get embeddings from the encoder
        patches = self.encoder(datacube)  # [B, L, D]

        # Calculate patch dimensions based on current input size
        H_patches = H_in // self.encoder.patch_size  # 512//8 = 64
        W_patches = W_in // self.encoder.patch_size  # 512//8 = 64
        
        # Reshape embeddings to [B, D, H', W']
        x = rearrange(patches, "B (H W) D -> B D H W", H=H_patches, W=W_patches)
        # x shape: [B, 1024, 64, 64]

        # Pass through convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 512, 64, 64]
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 512, 64, 64]
        x = self.conv_ps(x)  # [B, C_out * r^2, 64, 64] = [B, 64*64, 64, 64] = [B, 4096, 64, 64]

        # Upsample using PixelShuffle: [B, 4096, 64, 64] -> [B, 64, 512, 512]
        x = self.pixel_shuffle(x)  # [B, C_out, H_in, W_in] = [B, 64, 512, 512]

        # Final convolution to get desired output channels
        x = self.conv_out(x)  # [B, num_classes, 512, 512]

        return x