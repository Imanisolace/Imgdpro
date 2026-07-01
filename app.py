import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
from streamlit_image_comparison import image_comparison

st.set_page_config(page_title="AI Photo Restorer Pro", layout="wide")

st.title("🧹 AI Photo Restorer Pro")
st.write("Smart Auto: Detects faces, over-bright, over-sharp, and plastic skin. Fixes both ways.")

def analyze_image_smart(image):
    """Smart analysis that goes both ways and avoids plastic skin"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 1. Basic metrics
    noise_level = np.std(gray)
    brightness = np.mean(gray)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Face detection: if face present, be gentler
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    has_face = len(faces) > 0
    face_factor = 0.6 if has_face else 1.0 # reduce strength for faces
    
    # 3. Map metrics to slider values - NOW 2 WAY
    # Strength: More noise = more denoise. But cap for faces
    strength_raw = noise_level / 8
    strength = np.clip(strength_raw * face_factor, 2, 15)
    
    # Brightness: Dark = brighten, Bright = darken
    target_brightness = 120 # "ideal" mid gray
    bright_diff = target_brightness - brightness
    bright = np.clip(1.0 + (bright_diff / 255.0), 0.8, 1.5) # <1.0 means darken
    
    # Sharpness: Blurry = sharpen, Too Sharp = slight blur
    target_sharp = 200 # "ideal" sharpness
    sharp_diff = target_sharp - laplacian_var
    sharp = np.clip(2.0 + (sharp_diff / 200.0), 0, 4) # <1.0 means we will blur later
    
    # 4. Anti-plastic check: If strength is high AND face, reduce and add texture back
    anti_plastic = False
    if has_face and strength > 8:
        strength = 8 # cap denoise for skin
        sharp = max(sharp, 2.5) # add sharpness back to keep skin texture
        anti_plastic = True
        
    return strength, bright, sharp, has_face, anti_plastic

def denoise_image(image, h, bright_val, sharp_val):
    h = int(round(h))
    sharp_val = int(round(sharp_val))
    
    denoised = cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
    denoised = cv2.bilateralFilter(denoised, 5, 50, 50)
    denoised = cv2.convertScaleAbs(denoised, alpha=bright_val, beta=0)
    
    # Vice versa for sharpness
    if sharp_val > 1: # Sharpen
        kernel = np.array([[0, -1, 0], [-1, 4 + sharp_val, -1], [0, -1, 0]])
        denoised = cv2.filter2D(denoised, -1, kernel)
    elif sharp_val < 1 and sharp_val > 0: # Slight blur to reduce over-sharp halos
        denoised = cv2.GaussianBlur(denoised, (3,3), 0.5)
    
    return denoised

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
default_strength, default_bright, default_sharp = 7.0, 1.1, 2.0

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    col_btn1, col_btn2 = st.columns([1,4])
    with col_btn1:
        if st.button("✨ Smart Auto", use_container_width=True):
            s, b, sh, face, plastic = analyze_image_smart(image_bgr)
            st.session_state.strength = s
            st.session_state.bright = b
            st.session_state.sharp = sh
            
            msg = f"Auto set: Denoise={s:.1f}, Bright={b:.2f}, Sharp={sh:.1f}"
            if face: msg += " | Face detected"
            if plastic: msg += " | Anti-plastic applied"
            st.success(msg)

    st.sidebar.header("Edit Controls")
    strength = st.sidebar.slider("Noise Removal Strength", 1.0, 15.0, 
                                 st.session_state.get("strength", default_strength), 0.5)
    bright = st.sidebar.slider("Brightness Boost <1=Darken", 0.8, 1.5, 
                               st.session_state.get("bright", default_bright), 0.05)
    sharp = st.sidebar.slider("Sharpness <1=Deblur", 0.0, 4.0, 
                              st.session_state.get("sharp", default_sharp), 0.25)

    with st.spinner('Restoring photo...'):
        clean_image_bgr = denoise_image(image_bgr, strength, bright, sharp)
        clean_image_rgb = cv2.cvtColor(clean_image_bgr, cv2.COLOR_BGR2RGB)

    st.subheader("Before vs After")
    image_comparison(
        img1=image_np,
        img2=clean_image_rgb,
        label1="Before",
        label2="After",
        width=800
    )

    buf = io.BytesIO()
    result_pil = Image.fromarray(clean_image_rgb)
    result_pil.save(buf, format="PNG")
    st.download_button("📥 Download Restored Image", buf.getvalue(), "restored.png", "image/png")
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
