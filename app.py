import streamlit as st
import time

st.set_page_config(page_title="Mersenne Prime Finder", page_icon="🔢")

st.title("🔢 Mersenne Prime Finder - Lucas-Lehmer Test")
st.write("Finds primes of the form $M_p = 2^p - 1$ where p is prime")

def is_prime_small(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def lucas_lehmer(p):
    if p == 2:
        return True
    m = (1 << p) - 1
    s = 4
    for i in range(p - 2):
        s = (s * s - 2) % m
    return s == 0

col1, col2 = st.columns(2)
with col1:
    max_p = st.number_input("Max exponent p to test", min_value=2, max_value=2000, value=200, step=10)
with col2:
    run_btn = st.button("Find Mersenne Primes", type="primary")

if run_btn:
    progress = st.progress(0)
    status = st.empty()
    results = []
    
    primes_to_test = [p for p in range(2, max_p + 1) if is_prime_small(p)]
    total = len(primes_to_test)
    
    for idx, p in enumerate(primes_to_test):
        status.text(f"Testing p = {p}... {idx+1}/{total}")
        if lucas_lehmer(p):
            m = (1 << p) - 1
            digits = len(str(m))
            results.append((p, digits))
            st.success(f"Found! p = {p}, M_p has {digits} digits")
        progress.progress((idx + 1) / total)
        time.sleep(0.01) # so UI updates
    
    status.text("Done!")
    st.write("### All Mersenne Primes Found:")
    st.dataframe(results, column_config={"0": "Exponent p", "1": "Digits in M_p"})
    
st.caption("Note: Testing p > 1000 gets slow in the browser. For huge primes, use GIMPS.")    image = Image.open(uploaded_file).convert("RGB")
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
