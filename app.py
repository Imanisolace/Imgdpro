import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

st.set_page_config(page_title="AI Photo Restorer Pro", layout="wide")

st.title("🧹 AI Photo Restorer Pro")
st.write("Remove noise, brighten, and sharpen low-light photos. Powered by PDE-based filters.")
st.caption("Upload any JPG/PNG. No install needed.")

# --- Sidebar Controls ---
st.sidebar.header("Edit Controls")
strength = st.sidebar.slider("Noise Removal Strength", 1, 20, 10, help="Higher = more noise removed, but can get soft")
bright = st.sidebar.slider("Brightness Boost", 0.8, 1.5, 1.1, 0.05)
sharp = st.sidebar.slider("Sharpness", 0, 5, 2, help="Brings back details lost during denoising")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

def denoise_image(image, h, bright_val, sharp_val):
    """
    3-step PDE-based denoising pipeline
    1. NLM = Removes Gaussian noise
    2. Bilateral = Smooths while keeping edges
    3. Brighten + Sharpen = Make it look pro
    """
    # Step 1: Fast Non-Local Means. Main denoiser
    denoised = cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
    
    # Step 2: Bilateral Filter. Keep edges sharp
    denoised = cv2.bilateralFilter(denoised, 5, 50, 50)
    
    # Step 3a: Brightness/Contrast
    denoised = cv2.convertScaleAbs(denoised, alpha=bright_val, beta=0)
    
    # Step 3b: Unsharp Mask for detail
    if sharp_val > 0:
        kernel = np.array([[0, -1, 0], [-1, 4 + sharp_val, -1], [0, -1, 0]])
        denoised = cv2.filter2D(denoised, -1, kernel)
    
    return denoised

if uploaded_file is not None:
    # Load image and convert to OpenCV BGR format
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(image_np, use_column_width=True)

    with st.spinner('Restoring photo...'):
        clean_image_bgr = denoise_image(image_bgr, strength, bright, sharp)

    with col2:
        st.subheader("Restored")
        clean_image_rgb = cv2.cvtColor(clean_image_bgr, cv2.COLOR_BGR2RGB)
        st.image(clean_image_rgb, use_column_width=True)

    # Download button
    buf = io.BytesIO()
    result_pil = Image.fromarray(clean_image_rgb)
    result_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    
    st.download_button(
        label="📥 Download Restored Image",
        data=byte_im,
        file_name="restored.png",
        mime="image/png"
    )
    
    st.success("Done! Use the sliders to fine-tune the result.")

else:
    st.info("👆 Upload an image to get started")

st.sidebar.markdown("---")
st.sidebar.write("**Need custom image processing?**")
st.sidebar.write("DM for freelance work: Photo restoration, document cleanup, PDE simulations")
