from pathlib import Path
from io import BytesIO
from rasterio.io import MemoryFile
from pyproj import Geod

import numpy as np
import tifffile
import torch
from segmentation_models_pytorch import Unet

CHECKPOINT_PATH = Path("/home/titanx/Desktop/preprocessing/checkpoints/best_model.pth")
PATCH_SIZE = 256

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=2,
        classes=1,
        activation=None,
    ).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict)
    model.eval()
    return model, device

def read_tif_file(file_bytes: bytes):
    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            band0 = src.read(1).astype(np.float32)
            band1 = src.read(2).astype(np.float32)
            transform = src.transform
            crs = src.crs
    image = np.stack([band0, band1], axis=0)
    return image, band0, band1, transform, crs

def as_channels_first(image: np.ndarray) -> np.ndarray:
    image = np.squeeze(image)
    if image.ndim == 2:
        return image[np.newaxis, ...]
    if image.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D TIFF, received shape {image.shape}.")

    # TIFFs commonly arrive as (bands, height, width) or (height, width, bands).
    if image.shape[0] <= 32 and image.shape[1] > 32 and image.shape[2] > 32:
        return image
    if image.shape[2] <= 32 and image.shape[0] > 32 and image.shape[1] > 32:
        return np.moveaxis(image, -1, 0)
    raise ValueError(f"Could not identify the band axis for TIFF shape {image.shape}.")


def read_tiff(uploaded_file) -> np.ndarray:
    return as_channels_first(tifffile.imread(uploaded_file))


def convert_bands_to_npy(image: np.ndarray, band0: int, band1: int) -> np.ndarray:
    """Convert the selected TIFF bands to the same NumPy format used by training."""
    npy_bands = []
    for band_index in (band0, band1):
        buffer = BytesIO()
        np.save(buffer, image[band_index].astype(np.float32), allow_pickle=False)
        buffer.seek(0)
        npy_bands.append(np.load(buffer, allow_pickle=False))
    return np.stack(npy_bands, axis=0)

def predict(model, device, image):
    image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)
        mask = (probabilities >= 0.5).float().cpu().numpy().squeeze()
    return mask

def predict_mask(model, device, selected: np.ndarray, threshold: float):
    height, width = selected.shape[1:]
    probabilities = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)

    with torch.inference_mode():
        for y in range(0, height, PATCH_SIZE):
            for x in range(0, width, PATCH_SIZE):
                y_end = min(y + PATCH_SIZE, height)
                x_end = min(x + PATCH_SIZE, width)
                patch = selected[:, y:y_end, x:x_end]
                padded = np.zeros((2, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
                padded[:, : patch.shape[1], : patch.shape[2]] = patch
                tensor = torch.from_numpy(padded).unsqueeze(0).to(device)
                result = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
                probabilities[y:y_end, x:x_end] += result[: y_end - y, : x_end - x]
                counts[y:y_end, x:x_end] += 1

    probabilities /= np.maximum(counts, 1)
    mask = (probabilities >= threshold).astype(np.float32)
    return mask


def get_mask_longitude_bounds(mask: np.ndarray, transform):
    """Calculate leftmost and rightmost geographic coordinates of positive mask pixels."""
    positive_cols = np.where(mask > 0)[1]
    if len(positive_cols) == 0:
        return None, None
    min_col = int(positive_cols.min())
    max_col = int(positive_cols.max())
    leftmost_x = transform[2] + min_col * transform[0]
    rightmost_x = transform[2] + max_col * transform[0]
    return leftmost_x, rightmost_x


def get_mask_latitude_bounds(mask: np.ndarray, transform):
    """Calculate topmost and bottommost geographic coordinates of positive mask pixels."""
    positive_rows = np.where(mask > 0)[0]
    if len(positive_rows) == 0:
        return None, None
    min_row = int(positive_rows.min())
    max_row = int(positive_rows.max())
    # transform[5] = top-left y (latitude), transform[4] = pixel height (negative for north-up)
    topmost_y = transform[5] + min_row * transform[4]
    bottommost_y = transform[5] + max_row * transform[4]
    return topmost_y, bottommost_y


def calculate_spill_area(mask: np.ndarray, transform) -> float:
    """Calculate spill area in square meters using geodesic area calculation."""
    positive_pixels = np.where(mask > 0)
    if len(positive_pixels[0]) == 0:
        return 0.0
    
    # Get unique rows and columns of positive pixels
    rows = positive_pixels[0]
    cols = positive_pixels[1]
    
    # Calculate area by summing up pixel areas using geodesic calculation
    # For each row, calculate the latitude and sum up the area
    geod = Geod(ellps="WGS84")
    total_area = 0.0
    
    # Group by row to calculate area per row
    unique_rows = np.unique(rows)
    pixel_width = transform[0]  # longitude per pixel
    pixel_height = abs(transform[4])  # latitude per pixel (positive)
    
    for row in unique_rows:
        lat = transform[5] + row * transform[4]
        # Number of pixels in this row
        row_pixel_count = np.sum(rows == row)
        # Width in degrees at this latitude
        width_deg = row_pixel_count * pixel_width
        # Calculate geodesic area for this row's pixels
        # Using a polygon approximation for the row segment
        lon_start = transform[2] + cols[rows == row].min() * transform[0]
        lon_end = transform[2] + cols[rows == row].max() * transform[0] + pixel_width
        
        # Create a polygon for this row segment and calculate area
        poly_lons = [lon_start, lon_end, lon_end, lon_start, lon_start]
        poly_lats = [lat, lat, lat - pixel_height, lat - pixel_height, lat]
        area, _ = geod.polygon_area_perimeter(poly_lons, poly_lats)
        total_area += abs(area)
    
    return total_area


'''server starts'''
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager

# def calculate_mask_area_projected(mask: np.ndarray, transform):
#     pixel_width = abs(transform[0])
#     pixel_height = abs(transform[4])
#     pixel_area_m2 = pixel_width * pixel_height

#     positive_pixel_count = int(np.sum(mask > 0))
#     total_area_km2 = positive_pixel_count * pixel_area_m2 /1e6
#     return total_area_km2

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once on startup
    model, device = load_model()
    ml_models["model"] = model
    ml_models["device"] = device
    yield
    # Clean up on shutdown
    ml_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def run_prediction(file: UploadFile = File(...)):
    # Read binary stream as bytes
    file_bytes = await file.read()
    
    try:
        image, band0, band1, transform, crs = read_tif_file(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid TIFF file: {str(e)}")

    model = ml_models["model"]
    device = ml_models["device"]

    # Inference (selected shape is (2, H, W))
    mask = predict_mask(model, device, selected=image, threshold=0.5)
    positive_pixels = int(mask.sum())

    if positive_pixels == 0:
        return {
            "filename": file.filename,
            "shape": list(image.shape),
            "dtype": str(image.dtype),
            "positive_pixels": 0,
            "mean_probability": 0.0,
            "leftmost_longitude": None,
            "rightmost_longitude": None,
            "topmost_latitude": None,
            "bottommost_latitude": None,
            "crs": str(crs),
            "spill_area_sqm": 0.0,
            "spill_area_sqkm": 0.0,
            "is_spill": False
        }

    leftmost_x, rightmost_x = get_mask_longitude_bounds(mask, transform)
    topmost_y, bottommost_y = get_mask_latitude_bounds(mask, transform)
    spill_area_sqm = calculate_spill_area(mask, transform)

    return {
        "filename": file.filename,
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "positive_pixels": positive_pixels,
        "mean_probability": float(mask.mean()),
        "leftmost_longitude": leftmost_x,
        "rightmost_longitude": rightmost_x,
        "topmost_latitude": topmost_y,
        "bottommost_latitude": bottommost_y,
        "crs": str(crs),
        "spill_area_sqm": spill_area_sqm,
        "spill_area_sqkm": spill_area_sqm / 1_000_000,
        "is_spill": True
    }