'''
# Kelp Mapping with Clay Foundation Model

This repository is a fork of the [Clay Foundation Model](https://github.com/Clay-foundation/model), modified to perform semantic segmentation of kelp using Sentinel-2 satellite imagery. The project adapts the Clay model to map kelp in coastal regions, leveraging a custom dataset with Sentinel-2 bands and a tailored configuration for training and inference.

### License

- **Code and Model Weights**: Licensed under the [Apache License](LICENSE).
- **Documentation**: Licensed under the [CC-BY-4.0 License](https://creativecommons.org/licenses/by/4.0/).

### Project Overview

This project extends the Clay Foundation Model for binary segmentation of kelp (kelp vs. background) using Sentinel-2 imagery. Key modifications include:

- **Configuration**: Updated `configs/segment_kelp.yaml` to specify the data folder, test data folders, and band-specific mean/std values.
- **Data Module**: Modified `kelp_datamodule.py` to load Sentinel-2 bands and apply normalization.
- **Model**: Adapted `kelp_model.py` for kelp segmentation.
- **Inference**: Customized `kelp_inference.ipynb` to visualize predictions using a pre-trained checkpoint.
- **Segmentor**: Customized `factory.py` to work with 512 x 512 chips as well.

The model is trained and evaluated using PyTorch Lightning, optimized for Sentinel-2’s multispectral data.


## Data Preparation

The model requires Sentinel-2 imagery and corresponding binary masks for kelp segmentation.

### Dataset Structure

Organize your dataset as follows:
```text data/ ├── train/ │ ├── images/ # Sentinel-2 images (.tif) │ └── masks/ # Binary masks (.tif, 1=kelp, 0=background) ├── test_folder1/ # Test set 1 │ ├── images/ │ └── masks/ ├── test_folder2/ # Test set 2 │ ├── images/ │ └── masks/ ... ```
### Compute Mean and Std

Run the `calculate_mean_std.ipynb` notebook to compute mean and standard deviation for Sentinel-2 bands across the training dataset.


### Configure `segment_kelp.yaml`

Edit `configs/segment_kelp.yaml` to specify:

- **Data Folder**: Path to your dataset (e.g., `C:/data/kelp`).
- **Test Folders**: List of test folder names (e.g., `["test_folder1", "test_folder2"]`).
- **Normalization Parameters**: Mean and std values from `calculate_mean_std.ipynb`.

## Usage

### Training the Model

```bash
python claymodel/finetune/segment/segment.py fit --config configs/segment_kelp.yaml
```

### Running Inference on Test Dataset
Run kelp_inference.ipynb after making sure about the directories are set correctly in the notebook, e.g.,  the directory to the ckpt file.

### Running Inference on A Sentinel-2 Image
Run kelp_inference_S2fullscen.ipynb after you set the directory of the folder(s) (each) containing two files: B2B3B4B8.tif and B5B6B7B8aB11B12.tif.


## Acknowledgments

- Built upon the [Clay Foundation Model](https://github.com/Clay-foundation/model).
- Uses Sentinel-2 imagery from the [Copernicus Programme](https://scihub.copernicus.eu/).
- Powered by [PyTorch Lightning](https://lightning.ai/).
- Inspired by https://github.com/HakaiInstitute/clay-foundation-model/tree/oneclass-kelp/. 

## Contact

For questions or feedback, open an issue on GitHub.

