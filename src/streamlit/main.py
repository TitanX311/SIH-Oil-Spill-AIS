import streamlit as st
import torch
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
from pathlib import Path
import matplotlib.pyplot as plt

CHECKPOINT_PATH = Path("/home/titanx/Desktop/preprocessing/checkpoints/best_model.pth")

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=2,
        classes=1,
        activation=None,
    )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, device

def read_tif_file(uploaded_file):
    with rasterio.open(uploaded_file) as src:
        band0 = src.read(1).astype(np.float32)
        band1 = src.read(2).astype(np.float32)
    image = np.stack([band0, band1], axis=-1)
    return image, band0, band1

def read_npy_files(band0_file, band1_file):
    band0 = np.load(band0_file).astype(np.float32)
    band1 = np.load(band1_file).astype(np.float32)
    image = np.stack([band0, band1], axis=-1)
    return image, band0, band1

def predict(model, device, image):
    image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)
        mask = (probabilities >= 0.5).float().cpu().numpy().squeeze()
    return mask

def plot_image(band0, band1):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(band0, cmap="viridis")
    axes[0].set_title("Band 0 (VV)")
    axes[0].axis("off")
    axes[1].imshow(band1, cmap="viridis")
    axes[1].set_title("Band 1 (VH)")
    axes[1].axis("off")
    plt.tight_layout()
    return fig

def plot_mask(mask):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(mask, cmap="viridis")
    ax.set_title("Predicted Mask")
    ax.axis("off")
    plt.tight_layout()
    return fig

def main():
    st.set_page_config(page_title="Oil Spill Segmentation", layout="wide")
    st.title("Oil Spill Segmentation with U-Net (MobileNetV2)")

    model, device = load_model()
    st.success(f"Model loaded on {device}")

    input_mode = st.radio("Input Mode", ["TIFF file", "NPY files (band0 + band1)"], horizontal=True)

    if input_mode == "TIFF file":
        uploaded_file = st.file_uploader("Upload a .tif file", type=["tif", "tiff"])
        if uploaded_file is not None:
            image, band0, band1 = read_tif_file(uploaded_file)
            st.subheader("Uploaded Image")
            st.pyplot(plot_image(band0, band1))

            if st.button("Predict"):
                with st.spinner("Generating mask..."):
                    mask = predict(model, device, image)
                st.subheader("Predicted Mask")
                st.pyplot(plot_mask(mask))
                
                st.subheader("Overlay")
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax.imshow(band0, cmap="viridis")
                ax.imshow(mask, cmap="Reds", alpha=0.5)
                ax.set_title("Mask Overlay on Band 0")
                ax.axis("off")
                plt.tight_layout()
                st.pyplot(fig)

    else:
        col1, col2 = st.columns(2)
        with col1:
            band0_file = st.file_uploader("Upload band0.npy", type=["npy"])
        with col2:
            band1_file = st.file_uploader("Upload band1.npy", type=["npy"])

        if band0_file is not None and band1_file is not None:
            image, band0, band1 = read_npy_files(band0_file, band1_file)
            st.subheader("Uploaded Image")
            st.pyplot(plot_image(band0, band1))

            if st.button("Predict"):
                with st.spinner("Generating mask..."):
                    mask = predict(model, device, image)
                st.subheader("Predicted Mask")
                st.pyplot(plot_mask(mask))
                
                st.subheader("Overlay")
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax.imshow(band0, cmap="viridis")
                ax.imshow(mask, cmap="Reds", alpha=0.5)
                ax.set_title("Mask Overlay on Band 0")
                ax.axis("off")
                plt.tight_layout()
                st.pyplot(fig)

if __name__ == "__main__":
    main()