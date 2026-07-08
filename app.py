import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. PRO THEME + CONFIG
st.set_page_config(
    page_title="Heat Equation Pro",
    layout="wide",
    initial_sidebar_state="collapsed", # Collapsed helps on mobile
    page_icon="🔥"
)

# Custom CSS - COMPACT METRICS FOR PORTRAIT
st.markdown("""
<style>
.stApp {background-color: #0E1117; color: #FAFAFA;}
    h1, h2, h3 {color: #FF4B4B; font-size: 1.8rem;}
    [data-testid="stSidebar"] {background-color: #1A1C23;}
.stButton>button {background-color: #FF4B4B; color: white; border-radius: 8px; border: none; font-size: 14px;}
.stButton>button:hover {background-color: #FF6B6B;}
.metric-card {background-color: #262730; padding: 10px; border-radius: 8px; border: 1px solid #333;}

  /* COMPACT METRICS */
    [data-testid="stMetric"] {
        background-color: #1A1C23;
        padding: 8px 10px;
        border-radius: 8px;
        border: 1px solid #333;
        min-width: 0px!important; /* Allow shrinking */
    }
    [data-testid="stMetricValue"] {font-size: 1.2rem!important;} /* Smaller number */
    [data-testid="stMetricLabel"] {font-size: 0.8rem!important;} /* Smaller label */

  /* FORCE 2 COLUMNS DOWN TO 350px */
    [data-testid="column"] { min-width: 160px!important; }
</style>
""", unsafe_allow_html=True)

st.title("🔥 Heat Equation Pro v1.4.3")
st.caption("Professional 1D FDM Solver. Mobile-friendly dashboard.")

# SESSION STATE INIT
if 'alpha' not in st.session_state: st.session_state.alpha = 0.1
if 'L' not in st.session_state: st.session_state.L = 1.0
if 'has_run' not in st.session_state: st.session_state.has_run = False

# 2. HERO SECTION
if not st.session_state.has_run:
    st.markdown("### Get started in 3 seconds")
    st.info("👈 Step 1: Pick preset | Step 2: Click Solve")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔥 Metal Rod Demo", type="primary", use_container_width=True):
            st.session_state.alpha = 0.5; st.session_state.L = 1.0; st.session_state.has_run = True; st.rerun()
    with c2:
        if st.button("🧊 Ceramic Demo", use_container_width=True):
            st.session_state.alpha = 0.01; st.session_state.L = 2.0; st.session_state.has_run = True; st.rerun()
    st.divider()

# 3. SIDEBAR
with st.sidebar:
    st.header("⚙️ Simulation Setup")
    with st.expander("📚 What do these mean?", expanded=False):
        st.markdown("- **α**: Diffusivity\n- **L**: Length\n- **Nx**: Grid points\n- **r**: Must be < 0.5")

    st.subheader("⚡ Presets")
    c1, c2 = st.columns(2)
    if c1.button("Metal", use_container_width=True): st.session_state.alpha, st.session_state.L = 0.5, 1.0; st.rerun()
    if c2.button("Ceramic", use_container_width=True): st.session_state.alpha, st.session_state.L = 0.01, 2.0; st.rerun()

    alpha = st.slider("Diffusivity α", 0.001, 1.0, st.session_state.alpha, 0.001)
    L = st.number_input("Length L [m]", 0.1, 10.0, st.session_state.L)
    Nx = st.slider("Grid Nx", 50, 500, 200)
    T_final = st.slider("Final Time [s]", 0.1, 10.0, 1.0)
    bc_type = st.selectbox("BC", ["Dirichlet: u=0", "Neumann: Insulated"])

    st.divider()
    uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"])
    if uploaded_file: df_user = pd.read_csv(uploaded_file)

    dx = L / (Nx-1)
    dt = 0.49 * dx**2 / alpha
    Nt = int(T_final / dt)
    r = alpha * dt / dx**2
    st.info(f"**r:** {r:.3f} {'✅' if r < 0.5 else '❌'} | **Steps:** {Nt}")

# 4. MAIN
tab1, tab2, tab3 = st.tabs(["🔥 Heat Equation", "🎯 Root Finder 🔒", "📈 ODE Solver 🔒"])

with tab1:
    if st.button("▶️ Solve & Animate", type="primary", use_container_width=True):
        st.session_state.has_run = True; st.session_state.run_solver = True

    if st.session_state.get('run_solver', False) or st.session_state.has_run:
        x = np.linspace(0, L, Nx)
        u = np.sin(np.pi * x / L)
        frames = [u.copy()]

        progress_bar = st.progress(0, "Computing FDM...")
        for n in range(Nt):
            u_new = u.copy()
            u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
            u_new[0] = 0 if bc_type.startswith("Dirichlet") else u_new[1]
            u_new[-1] = 0 if bc_type.startswith("Dirichlet") else u_new[-2]
            u = u_new
            if n % max(1, Nt//100) == 0: frames.append(u.copy())
            progress_bar.progress((n+1)/Nt)
        progress_bar.empty()
        st.success("Done!")
        st.session_state.run_solver = False

        # 1. METRICS - COMPACT 2x2 GRID
        st.markdown("### 📊 Results")
        col1, col2 = st.columns(2, gap="small") # small gap for mobile

        loss = (max(frames[0])-max(frames[-1]))*100

        with col1:
            st.metric(label="🌡️ Max @ t=0", value=f"{max(frames[0]):.3f}", delta="°C")
            st.metric(label="📉 Loss", value=f"{loss:.1f}%", delta=f"-{loss:.1f}%", delta_color="inverse")

        with col2:
            st.metric(label="❄️ Max @ t=final", value=f"{max(frames[-1]):.3f}", delta="°C")
            st.metric(label="⏱️ Time", value=f"{T_final:.2f}s")

        st.divider()

        # 2. PLOT
        col_plot, col_download = st.columns([4, 1])
        with col_plot:
            st.subheader("📈 Animation")
            plot_spot = st.empty()
            for i, frame in enumerate(frames):
                fig, ax = plt.subplots(figsize=(8,3), facecolor="#0E1117") # Smaller fig for mobile
                ax.plot(x, frame, color="#FF4B4B", linewidth=2)
                if uploaded_file: ax.plot(df_user['x'], df_user['u_final'], 'b--', label="Your Data")
                ax.set_facecolor("#1A1C23")
                ax.tick_params(colors='white', which='both', labelsize=8)
                ax.yaxis.label.set_color('white'); ax.xaxis.label.set_color('white')
                ax.set_ylim(-0.1, 1.1)
                ax.set_xlabel("x [m]", fontsize=9)
                ax.set_ylabel("u(x,t)", fontsize=9)
                ax.set_title(f"t: {i*dt*(Nt//100):.3f} s", color="white", fontsize=10)
                ax.legend(facecolor="#262730", edgecolor="white", labelcolor='white', fontsize=8)
                ax.grid(True, alpha=0.2, color='gray')
                plot_spot.pyplot(fig)
                plt.close(fig)

        with col_download:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("Export")
            df = pd.DataFrame({"x": x, "u_final": frames[-1]})
            csv = df.to_csv(index=False)
            st.download_button("📥 CSV", csv, "heat_solution.csv", use_container_width=True)
            st.button("📄 PDF - Pro", use_container_width=True, disabled=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("Click 'Solve & Animate' to see results")

with tab2:
    st.subheader("🎯 Root Finder")
    st.button("🔒 Unlock Pro $19/mo", disabled=True)
with tab3:
    st.subheader("📈 ODE Solver")
    st.button("🔒 Unlock Pro $19/mo", disabled=True)