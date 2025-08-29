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
- **Segmentor**: Customized `factory.py` to work with 512 x 512 chips.

The model is trained and evaluated using PyTorch Lightning, optimized for Sentinel-2’s multispectral data.

## Getting Started

### Prerequisites

- **Python**: Version >= 3.8.
- **Hardware**: GPU (CUDA-enabled) recommended for training.
- **Dependencies**: Managed via `environment.yml` or `pip`.
- **JupyterLab**: Required for running the inference notebook.
- **Anaconda/Mamba**: Recommended for environment management.

### Installation

#### Option 1: Pip Installation (Recommended)

Install the modified `claymodel` package and dependencies:

```bash
pip install git+https://github.com/<your-username>/<your-repo>.git
```

#### Option 2: Development Installation

For development or customization, clone the repository and set up the environment:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. **Install Mamba**:

   Follow the [Mamba installation guide](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) for faster dependency resolution.

3. **Create and Activate Environment**:

   ```bash
   mamba env create --file environment.yml
   mamba activate claymodel-env
   ```

4. **Verify Dependencies**:

   ```bash
   mamba list
   ```

5. **Install Jupyter Kernel**:

   ```bash
   python -m ipykernel install --user --name claymodel-env
   jupyter kernelspec list --json
   ```

## Data Preparation

The model requires Sentinel-2 imagery and corresponding binary masks for kelp segmentation.

### Dataset Structure

Organize your dataset as follows:
data/
├── train/
│   ├── images/  # Sentinel-2 images (.tif)
│   └── masks/   # Binary masks (.tif, 1=kelp, 0=background)
├── test_folder1/  # Test set 1
│   ├── images/
│   └── masks/
├── test_folder2/  # Test set 2
│   ├── images/
│   └── masks/
...
text### Compute Mean and Std

1. Run the `calculate_mean_std.ipynb` notebook to compute mean and standard deviation for Sentinel-2 bands across the training dataset.
2. Example output for 5 bands (replace with your actual values):

   ```python
   mean = [0.1234, 0.2345, 0.3456, 0.4567, 0.5678]
   std = [0.0123, 0.0234, 0.0345, 0.0456, 0.0567]
   ```

### Configure `segment_kelp.yaml`

Edit `configs/segment_kelp.yaml` to specify:

- **Data Folder**: Path to your dataset (e.g., `C:/data/kelp`).
- **Test Folders**: List of test folder names (e.g., `["test_folder1", "test_folder2"]`).
- **Normalization Parameters**: Mean and std values from `calculate_mean_std.ipynb`.

## Usage

### Training the Model

1. **Activate Environment**:

   ```bash
   mamba activate claymodel-env
   cd C:/clay/clay-model
   ```

2. **Run Training**:

   ```bash
   python claymodel/finetune/segment/segment.py fit --config configs/segment_kelp.yaml
   ```

   This trains the model and evaluates it on the test folders. Checkpoints are saved in `lightning_logs/`.

3. **Fast Development Run** (Optional):

   Test on one batch:

   ```bash
   python claymodel/finetune/segment/segment.py fit --config configs/segment_kelp.yaml --trainer.fast_dev_run=True
   ```

### Running Inference

1. **Launch JupyterLab**:

   ```bash
   mamba activate claymodel-env
   cd C:/clay/clay-model
   jupyter lab
   ```

2. **Modify `kelp_inference.ipynb`**:

   - Open `kelp_inference.ipynb` in JupyterLab.
   - Update the checkpoint path to your trained model’s `.ckpt` file.
   - Set the output directory for visualizations.
   - Example:

     ```python
     CHECKPOINT_PATH = "C:/clay/clay-model/lightning_logs/version_0/checkpoints/epoch=XX.ckpt"
     OUTPUT_DIR = "C:/Users/<your-username>/saved_models/AnnotatedData/LeaveOneOutRev/kelp"
     ```

3. **Run Inference**:

   Execute the notebook to generate visualizations, including false-color images, ground truth masks, and predicted masks saved as PNGs.

## Model Details

### Data Module

- **File**: `kelp_datamodule.py`
- **Purpose**: Loads Sentinel-2 bands (B2, B3, B4, B8, B5) and applies normalization using the specified mean/std values.

### Model

- **File**: `kelp_model.py`
- **Purpose**: Defines the segmentation model, fine-tuned for binary kelp segmentation.

### Inference

- **File**: `kelp_inference.ipynb`
- **Purpose**: Visualizes predictions using false-color composites (NIR-Red-Green) and overlays kelp boundaries.

## Results

The model outputs:

- **False-Color Images**: NIR-Red-Green composites (bands B8, B4, B3).
- **Ground Truth Masks**: Binary kelp masks.
- **Predicted Masks**: Model predictions with kelp pixels overlaid.
- **Metrics**:
  - Per-image IoU
  - Dataset IoU
  - Precision
  - Recall
  - F1-Score

Example metrics (replace with your results):
=== TEST SET RESULTS ===
Kelp Per-Image IoU: 0.XXXX ± 0.XXXX
Kelp Dataset IoU: 0.XXXX
Kelp Precision: 0.XXXX
Kelp Recall: 0.XXXX
Kelp F1-Score: 0.XXXX
Total test samples: XXX
text## Contributing


## Acknowledgments

- Built upon the [Clay Foundation Model](https://github.com/Clay-foundation/model).
- Uses Sentinel-2 imagery from the [Copernicus Programme](https://scihub.copernicus.eu/).
- Powered by [PyTorch Lightning](https://lightning.ai/).

## Contact

For questions or feedback, open an issue on GitHub.

