"""
Command line interface to run the neural network model for Kelp classification!
From the project root directory, do:
    python segment.py fit --config configs/segment_kelp.yaml
References:
- https://lightning.ai/docs/pytorch/2.1.0/cli/lightning_cli.html
- https://pytorch-lightning.medium.com/introducing-lightningcli-v2-supercharge-your-training-c070d43c7dd6
"""
from lightning.pytorch.cli import LightningCLI
from claymodel.finetune.segment.kelp_datamodule import (
    KelpDataModule,  # noqa: F401
)
from claymodel.finetune.segment.kelp_model import (
    KelpSegmentor,  # noqa: F401
)
# %%
def cli_main():
    """
    Command-line interface to run Segmentation Model with KelpDataModule and KelpSegmentor.
    """
    cli = LightningCLI(
        KelpSegmentor,     # Custom Kelp binary segmentation model
        KelpDataModule,    # Custom Kelp data module with leave-one-folder-out
        save_config_kwargs={"overwrite": True},
    )
    return cli
# %%
if __name__ == "__main__":
    cli_main()
    print("Done!")