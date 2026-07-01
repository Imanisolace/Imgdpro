import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

st.set_page_config(page_title="Image Denoiser", layout="centered")
st.title("🧹 Image Denoising Tool")
st.write("Upload a noisy image. I’ll remove Gaussian + Salt & Pepper noise.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

def denoise_image(image):
    denoised_nlm = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    denoised_bilateral = cv2.bilateralFilter(denoised_nlm, 9, 75, 75)
    denoised_final = cv2.medianBlur(denoised_bilateral, 3)
    return denoised_final

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

    with st.spinner('Denoising...'):
        clean_image = denoise_image(image)

    with col2:
        st.subheader("Denoised")
        st.image(cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB), use_column_width=True)

    result = Image.fromarray(cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    st.download_button("Download Clean Image", buf.getvalue(), "denoised.png", "image/png")

st.sidebar.markdown("---")
st.sidebar.write("**Want custom image processing?**")
st.sidebar.write("DM for freelance work")
