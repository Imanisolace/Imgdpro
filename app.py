import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. PRO THEME + CONFIG
st.set_page_config(
    page_title="Heat Equation Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔥"
)

# Custom CSS
st.markdown("""
<style>
   .stApp {background-color: #0E1117; color: #FAFAFA;}
    h1, h2, h3 {color: #FF4B4B;}
    [data-testid="stSidebar"] {background-color: #1A1C23;}
   .stButton>button {background-color: #FF4B4B; color: white; border-radius: 8px; border: none;}
   .stButton>button:hover {background-color: #FF6B6B;}
   .metric-card {background-color: #262730; padding: 15px; border-radius: 8px; border: 1px solid #333;}
</style>
""", unsafe_allow_html=True)

st.title("🔥 Heat Equation Pro v1.3")
st.caption("Professional 1D FDM Solver with Stability Control. Export-ready plots.")

# SESSION STATE INIT
if 'alpha' not in st.session_state: st.session_state.alpha = 0.1
if 'L' not in st.session_state: st.session_state.L = 1.0
if 'has_run' not in st.session_state: st.session_state.has_run = False

# 2. HERO SECTION FOR NEW USERS - THEY CAN'T MISS THIS
if not st.session_state.has_run:
    st.markdown("### Get started in 3 seconds")
    st.info("👈 Step 1: Pick a preset in the sidebar | Step 2: Click Solve below")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔥 Run Metal Rod Demo", type="primary", use_container_width=True):
            st.session_state.alpha = 0.5
            st.session_state.L = 1.0
            st.session_state.has_run = True
            st.rerun()
    with c2:
        if st.button("🧊 Run Ceramic Demo", use_container_width=True):
            st.session_state.alpha = 0.01
            st.session_state.L = 2.0
            st.session_state.has_run = True
            st.rerun()
    with c3:
        st.button("📁 Upload CSV", use_container_width=True, disabled=True, help="Compare your data in Pro")
    st.divider()

# 3. SIDEBAR: INPUTS + PRESETS + UPLOAD
with st.sidebar:
    st.header("⚙️ Simulation Setup")

    with st.expander("📚 What do these mean?", expanded=False):
        st.markdown("""
        - **α**: Thermal diffusivity. Bigger = heats up faster. Metal=0.5, Ceramic=0.01
        - **L**: Length of your rod in meters
        - **Nx**: Grid points. 200 is good. 500 is for research
        - **r**: Stability number. Must be < 0.5 or sim crashes. We auto-calc it
        """)

    st.subheader("⚡ Material Presets")
    c1, c2 = st.columns(2)
    if c1.button("Metal Rod", use_container_width=True):
        st.session_state.alpha, st.session_state.L = 0.5, 1.0
        st.rerun()
    if c2.button("Ceramic", use_container_width=True):
        st.session_state.alpha, st.session_state.L = 0.01, 2.0
        st.rerun()

    alpha = st.slider("Diffusivity α", 0.001, 1.0, st.session_state.alpha, 0.001)
    L = st.number_input("Domain Length L [m]", 0.1, 10.0, st.session_state.L)
    Nx = st.slider("Grid Points Nx", 50, 500, 200)
    T_final = st.slider("Final Time [s]", 0.1, 10.0, 1.0)
    bc_type = st.selectbox("Boundary Conditions", ["Dirichlet: u=0", "Neumann: Insulated"])

    st.divider()
    st.subheader("📁 Compare Data")
    uploaded_file = st.file_uploader("Upload CSV to compare", type=["csv"])
    if uploaded_file:
        df_user = pd.read_csv(uploaded_file)

    dx = L / (Nx-1)
    dt = 0.49 * dx**2 / alpha # Keep r < 0.5 for stability
    Nt = int(T_final / dt)
    r = alpha * dt / dx**2

    st.info(f"**Stability:** r = {r:.3f} {'✅' if r < 0.5 else '❌'} \n**Steps:** {Nt} \n**dt:** {dt:.5f}")

# 4. MAIN AREA: PLOT + RESULTS
tab1, tab2, tab3 = st.tabs(["🔥 Heat Equation", "🎯 Root Finder 🔒", "📈 ODE Solver 🔒"])

with tab1:
    # TOP SOLVE BUTTON - CAN'T MISS IT
    if st.button("▶️ Solve & Animate", type="primary", use_container_width=True, key="top_solve"):
        st.session_state.has_run = True
        st.session_state.run_solver = True

    col_plot, col_data = st.columns([3, 1])

    if st.session_state.get('run_solver', False) or st.session_state.has_run:
        x = np.linspace(0, L, Nx)
        u = np.sin(np.pi * x / L) # IC
        frames = [u.copy()]

        progress_bar = st.progress(0, "Computing FDM...")
        # FDM SOLVER
        for n in range(Nt):
            u_new = u.copy()
            u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
            u_new[0] = 0 if bc_type.startswith("Dirichlet") else u_new[1]
            u_new[-1] = 0 if bc_type.startswith("Dirichlet") else u_new[-2]
            u = u_new
            if n % max(1, Nt//100) == 0: frames.append(u.copy())
            progress_bar.progress((n+1)/Nt)
        progress_bar.empty()
        st.success("Simulation Complete!")
        st.session_state.run_solver = False

        # ANIMATION
        with col_plot:
            st.subheader("Temperature Diffusion")
            plot_spot = st.empty()
            for i, frame in enumerate(frames):
                fig, ax = plt.subplots(figsize=(9,4), facecolor="#0E1117")
                ax.plot(x, frame, color="#FF4B4B", linewidth=2.5, label=f"t={i*dt*(Nt//100):.3f}s")
                if uploaded_file: ax.plot(df_user['x'], df_user['u_final'], 'b--', label="Your Data")
                ax.set_facecolor("#1A1C23")
                ax.tick_params(colors='white')
                ax.set_ylim(-0.1, 1.1)
                ax.set_xlabel("x [m]", color="white")
                ax.set_ylabel("u(x,t)", color="white")
                ax.set_title(f"Time: {i*dt*(Nt//100):.3f} s", color="white")
                ax.legend(facecolor="#262730", edgecolor="white")
                ax.grid(True, alpha=0.2)
                plot_spot.pyplot(fig)
                plt.close(fig)

        # RESULTS CARD
        with col_data:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Max Temp @ t=0", f"{max(frames[0]):.4f}")
            st.metric("Max Temp @ t=final", f"{max(frames[-1]):.4f}")
            st.metric("Total Energy Loss", f"{(max(frames[0])-max(frames[-1]))*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)

            df = pd.DataFrame({"x": x, "u_final": frames[-1]})
            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "heat_solution.csv", use_container_width=True)

            st.button("📄 Export PDF Report - Pro", use_container_width=True, disabled=True)

    else:
        with col_plot:
            st.info("Click 'Run Demo' above or 'Solve & Animate' to see results here")

with tab2:
    st.subheader("🎯 Root Finder")
    st.write("Newton-Raphson, Secant, Bisection with convergence plots.")
    st.button("🔒 Unlock in Pro Plan $19/mo", disabled=True, key="root_unlock")

with tab3:
    st.subheader("📈 ODE Solver")
    st.write("RK4, Adaptive Euler with error estimation.")
    st.button("🔒 Unlock in Pro Plan $19/mo", disabled=True, key="ode_unlock")