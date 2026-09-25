# Oil Spill Segmentation Model and API

This project trains and serves a deep learning model for oil spill detection from satellite imagery using a U-Net segmentation network. The model consumes dual-band input patches, typically VV and VH radar channels, and predicts a binary mask highlighting oil spill regions.

## Overview

- Model architecture: U-Net with MobileNetV2 encoder
- Input channels: 2 (VV + VH)
- Output: 1-channel binary segmentation mask
- Framework: PyTorch + segmentation-models-pytorch
- API layer: FastAPI
- Interactive UI: Streamlit

## Project Structure

```text
.
├── checkpoints/
│   └── best_model.pth
├── scripts/
│   ├── evaluate.py
│   ├── preprocess.py
│   ├── train.py
│   ├── verify.py
│   └── visualize.py
├── src/
│   ├── common.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── instance.py
│   ├── image_processing_server/
│   │   ├── app.py
│   │   └── sample_request.py
│   ├── preprocessing/
│   │   └── patch_gen.py
│   └── streamlit/
│       └── main.py
├── processed/                  # generated after preprocessing
├── preview/                    # generated during preprocessing
└── README.md
```

## Model Details

The segmentation model is defined in:

- [scripts/train.py](scripts/train.py)
- [scripts/evaluate.py](scripts/evaluate.py)
- [src/image_processing_server/app.py](src/image_processing_server/app.py)

The model is created as:

```python
Unet(
    encoder_name="mobilenet_v2",
    encoder_weights=None,
    in_channels=2,
    classes=1,
    activation=None,
)
```

It uses a binary segmentation objective with:

- BCEWithLogitsLoss
- Dice loss
- Combined BCE + Dice loss for training

A threshold of 0.5 is used for final mask generation.

## Data Pipeline

The preprocessing pipeline creates patch-level training samples from TIFF images and masks.

### Expected dataset layout

```text
processed/
├── band0/
│   └── *.npy
├── band1/
│   └── *.npy
├── masks/
│   └── *.npy
```

During preprocessing, the project:

- reads source image and mask pairs
- augments them with random flips, rotations, and crop operations
- saves band-specific arrays as `.npy`
- stores patches in `processed/`

## Training Workflow

### 1) Preprocess data

```bash
python scripts/preprocess.py
```

This generates training patches under `./processed`.

### 2) Train the model

```bash
python scripts/train.py
```

Configuration in this script includes:

- `EPOCHS = 10`
- `BATCH_SIZE = 2`
- `LEARNING_RATE = 1e-4`
- `WEIGHT_DECAY = 1e-4`
- `VAL_PERCENT = 0.0005`
- `SEED = 42`

The best checkpoint is saved to:

```text
checkpoints/best_model.pth
```

### 3) Evaluate the model

```bash
python scripts/evaluate.py
```

This loads the saved checkpoint, runs evaluation on a subset of processed samples, and reports segmentation metrics.

## Server

The FastAPI server is implemented in:

- [src/image_processing_server/app.py](src/image_processing_server/app.py)

It loads the trained checkpoint at startup and exposes a prediction endpoint.

### Start the server

```bash
uvicorn src.image_processing_server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Prediction endpoint

```http
POST /predict
```

Request:

- form-data file field named `file`
- input should be a TIFF image

Example response:

```json
{
  "filename": "sample.tif",
  "shape": [2, 1024, 1024],
  "dtype": "float32",
  "positive_pixels": 23142,
  "mean_probability": 0.1831,
  "leftmost_longitude": 123.456,
  "rightmost_longitude": 123.789,
  "topmost_latitude": 45.678,
  "bottommost_latitude": 45.543,
  "crs": "EPSG:4326",
  "spill_area_sqm": 1284200.45,
  "spill_area_sqkm": 1.2842,
  "is_spill": true
}
```

The API computes:

- predicted segmentation mask
- bounding coordinates of detected spill region
- estimated spill area using geodesic area calculation
- boolean spill detection flag

### Sample request

A sample client request is available in:

- [src/image_processing_server/sample_request.py](src/image_processing_server/sample_request.py)

```bash
python src/image_processing_server/sample_request.py
```

## Streamlit Demo

The UI for image upload and mask visualization is in:

- [src/streamlit/main.py](src/streamlit/main.py)

Run it with:

```bash
streamlit run src/streamlit/main.py
```

This app allows uploading:

- TIFF files
- `.npy` band files (`band0.npy`, `band1.npy`)

and displays the predicted oil spill mask.

## Environment and Dependencies

Typical dependencies for this project include:

- Python 3.10+
- PyTorch
- segmentation-models-pytorch
- numpy
- rasterio
- tifffile
- albumentations
- FastAPI
- uvicorn
- streamlit
- pyproj

Install dependencies with your preferred environment manager, for example:

```bash
pip install torch torchvision segmentation-models-pytorch numpy rasterio tifffile albumentations fastapi uvicorn streamlit pyproj
```

## Notes

- The model path is configured in the training and inference scripts, so ensure your checkpoint exists at `checkpoints/best_model.pth` before running inference or the server.
- The server expects TIFF inputs with raster bands compatible with the trained model structure.
- The preprocessing stage uses patch-based augmentation suitable for satellite segmentation tasks.

## Typical Workflow

```bash
# 1. Prepare patches
python scripts/preprocess.py

# 2. Train model
python scripts/train.py

# 3. Start API server
uvicorn src.image_processing_server.app:app --host 0.0.0.0 --port 8000

# 4. Open the UI (optional)
streamlit run src/streamlit/main.py
```

## License

This project is for research and internal application use unless a separate license is provided by the repository owner.
