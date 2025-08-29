"""
LightningModule for training and validating a kelp segmentation model using the
Segmentor class with binary classification (sigmoid + threshold).
"""

import lightning as L
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import optim
from torchmetrics.classification import F1Score, BinaryJaccardIndex

from claymodel.finetune.segment.factory import Segmentor


class KelpSegmentor(L.LightningModule):
    """
    LightningModule for binary kelp segmentation tasks, utilizing Clay Segmentor.

    Attributes:
        model (nn.Module): Clay Segmentor model.
        loss_fn (nn.Module): The loss function (BCE with logits).
        iou (Metric): Intersection over Union metric for binary classification.
        f1 (Metric): F1 Score metric for binary classification.
        threshold (float): Threshold for binary predictions (default: 0.5).
    """

    def __init__(  # # noqa: PLR0913
        self,
        num_classes,
        ckpt_path,
        lr=1e-4,
        wd=0.05,
        b1=0.9,
        b2=0.95,
        threshold=0.5,
    ):
        super().__init__()
        self.save_hyperparameters()  # Save hyperparameters for checkpointing
        
        # Binary classification: 1 output channel
        self.model = Segmentor(
            num_classes=num_classes,  # Single output for binary classification
            ckpt_path=ckpt_path,
        )

        # Use BCE with logits loss for binary segmentation
        # self.loss_fn = torch.nn.BCEWithLogitsLoss() 
        self.loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True) #smp.losses.DiceLoss(mode="binary") by default expects probabilities (values between 0 and 1)
                                                                                # The Dice loss formula requires probability values to compute the intersection and union correctly
        
        # Binary metrics
        self.iou = BinaryJaccardIndex(threshold=threshold) # This will compute IoU across the entire dataset rather than averaging per-image IoUs.
        self.f1 = F1Score(
            task="binary",
            threshold=threshold,
        )
        
        self.threshold = threshold

    def forward(self, datacube):
        """
        Forward pass through the segmentation model.

        Args:
            datacube (dict): A dictionary containing the input datacube and
                meta information like time, latlon, gsd & wavelengths.

        Returns:
            torch.Tensor: The segmentation logits (before sigmoid).
        """
        # Create Sentinel-2 wavelengths (in micrometers) based on the bands used. Look at kelp_datamodule file: load_module method from KelpDataset class
        # Based on your band structure: B2,B3,B4,B8,B5,B6,B7,B8A,B11,B12, here are the wavelength values derived from metadata
        # B2 or Blue: 0.493, B3 or green: 0.56, B4 or red: 0.665, B8 or nir: 0.842  B5 or rededge1: 0.704, B6 or rededge2: 0.74, B7 or rededge3: 0.783, B8A or nir08: 0.865, B11 or swir16: 1.61, B12 or swir22: 2.19
        waves = torch.tensor([0.493, 0.56, 0.665, 0.842, 0.704, 0.74, 0.783, 0.865, 1.61, 2.19]) 
        gsd = torch.tensor(10.0)  # Sentinel-2 10m GSD for these bands

        
        # Forward pass through the network
        return self.model(
            {
                "pixels": datacube["pixels"],
                "time": datacube["time"],
                "latlon": datacube["latlon"],
                "gsd": gsd,
                "waves": waves,
            },
        )

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer and scheduler
            configuration.
        """
        optimizer = optim.AdamW(
            [
                param
                for name, param in self.model.named_parameters()
                if param.requires_grad
            ],
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
            betas=(self.hparams.b1, self.hparams.b2),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=100,
            T_mult=1,
            eta_min=self.hparams.lr * 0.01,  # Fixed: was multiplying by 100
            last_epoch=-1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def shared_step(self, batch, batch_idx, phase):
        """
        Shared step for training and validation.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.
            phase (str): The phase (train or val).

        Returns:
            torch.Tensor: The loss value.
        """
        labels = batch["label"].float()  # Convert to float for loss
        outputs = self(batch)  # Shape: (batch_size, 1, H, W)
        
        # Resize outputs to match label size (512x512 to match your target_size)
        if outputs.shape[-2:] != labels.shape[-2:]:
            outputs = F.interpolate(
                outputs,
                size=labels.shape[-2:],  # Match the label dimensions
                mode="bilinear",
                align_corners=False,
            )
        
        # Squeeze the channel dimension for loss calculation: (batch_size, H, W)
        outputs_squeezed = outputs.squeeze(1)
        
        # Calculate loss
        loss = self.loss_fn(outputs_squeezed, labels)
        
        # Apply sigmoid and threshold for metrics
        probs = torch.sigmoid(outputs_squeezed)
        predictions = (probs > self.threshold).float()
        
        # Calculate metrics
        iou = self.iou(predictions, labels.int())
        f1 = self.f1(predictions, labels.int())

        # Log metrics
        self.log(
            f"{phase}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{phase}/iou",
            iou,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{phase}/f1",
            f1,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss value.
        """
        return self.shared_step(batch, batch_idx, "val")

    def predict_step(self, batch, batch_idx):
        """
        Prediction step for inference.

        Args:
            batch (dict): A dictionary containing the batch data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: Binary predictions (0 or 1).
        """
        outputs = self(batch)
        
        # Apply sigmoid and threshold
        probs = torch.sigmoid(outputs.squeeze(1))
        predictions = (probs > self.threshold).float()
        
        return predictions