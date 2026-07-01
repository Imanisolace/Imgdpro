import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
from streamlit_image_comparison import image_comparison

st.set_page_config(page_title="AI Photo Restorer Pro", layout="wide")

st.title("🧹 AI Photo Restorer Pro")
st.write("1-Click AI restoration with pro manual controls. Powered by PDE-based filters.")
st.caption("Upload any JPG/PNG. No install needed.")

def analyze_image(image):
    """Auto-analyze image and return suggested slider values"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Noise level: std dev of image. Higher = more noise
    noise_level = np.std(gray)
    
    # 2. Brightness: average pixel value 0-255
    brightness = np.mean(gray)
    
    # 3. Sharpness: variance of Laplacian. Lower = more blurry
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Map metrics to slider values
    strength = np.clip(noise_level / 8, 3, 15)  # More noise = more denoise
    bright = np.clip(1.8 - (brightness / 255.0), 0.9, 1.5)  # Darker = brighter
    sharp = np.clip(5.0 - (laplacian_var / 100.0), 1, 4)  # Blurrier = more sharp
    
    return strength, bright, sharp

def denoise_image(image, h, bright_val, sharp_val):
    # Cast floats from slider to int for OpenCV
    h = int(round(h))
    sharp_val = int(round(sharp_val))
    
    # Step 1: Fast Non-Local Means. Main denoiser
    denoised = cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
    
    # Step 2: Bilateral Filter. Keep edges sharp
    denoised = cv2.bilateralFilter(denoised, 5, 50, 50)
    
    # Step 3a: Brightness/Contrast. alpha can be float
    denoised = cv2.convertScaleAbs(denoised, alpha=bright_val, beta=0)
    
    # Step 3b: Unsharp Mask for detail
    if sharp_val > 0:
        kernel = np.array([[0, -1, 0], [-1, 4 + sharp_val, -1], [0, -1, 0]])
        denoised = cv2.filter2D(denoised, -1, kernel)
    
    return denoised

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Default slider values
default_strength, default_bright, default_sharp = 7.0, 1.1, 2.0

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # Auto button
    col_btn1, col_btn2 = st.columns([1,4])
    with col_btn1:
        if st.button("✨ Auto Enhance", use_container_width=True):
            s, b, sh = analyze_image(image_bgr)
            st.session_state.strength = s
            st.session_state.bright = b
            st.session_state.sharp = sh
            st.success(f"Auto set: Denoise={s:.1f}, Bright={b:.2f}, Sharp={sh:.1f}")

    # --- Sidebar Controls ---
    st.sidebar.header("Edit Controls")
    strength = st.sidebar.slider("Noise Removal Strength", 1.0, 15.0, 
                                 st.session_state.get("strength", default_strength), 0.5)
    bright = st.sidebar.slider("Brightness Boost", 0.8, 1.5, 
                               st.session_state.get("bright", default_bright), 0.05)
    sharp = st.sidebar.slider("Sharpness", 0.0, 4.0, 
                              st.session_state.get("sharp", default_sharp), 0.25)

    with st.spinner('Restoring photo...'):
        clean_image_bgr = denoise_image(image_bgr, strength, bright, sharp)
        clean_image_rgb = cv2.cvtColor(clean_image_bgr, cv2.COLOR_BGR2RGB)

    st.subheader("Before vs After")
    image_comparison(
        img1=image_np, # Original
        img2=clean_image_rgb, # Restored
        label1="Before",
        label2="After",
        width=800
    )

    # Download button
    buf = io.BytesIO()
    result_pil = Image.fromarray(clean_image_rgb)
    result_pil.save(buf, format="PNG")
    
    st.download_button(
        label="📥 Download Restored Image",
        data=buf.getvalue(),
        file_name="restored.png",
        mime="image/png",
        use_container_width=True
    )

else:
    st.info("👆 Upload an image to get started")

st.sidebar.markdown("---")
st.sidebar.write("**Need bulk photo restoration?**")
st.sidebar.write("DM for freelance PDE image processing")
