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

# Custom CSS for "Engineer Software" look
st.markdown("""
<style>
    .stApp {background-color: #0E1117; color: #FA;}
    h1, h2, h3 {color: #FF4B4B;}
    [data-testid="stSidebar"] {background-color: #1A1C23;}
    .stButton>button {background-color: #FF4B4B; color: white; border-radius: 8px;}
    .metric-card {background-color: #262730; padding: 10px; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

st.title("🔥 Heat Equation Pro v1.2")
st.caption("Professional 1D FDM Solver with Stability Control. Export-ready plots.")

# 2. SIDEBAR: INPUTS + PRESETS + UPLOAD
with st.sidebar:
    st.header("⚙️ Simulation Setup")
    
    st.subheader("⚡ Quick Presets")
    c1, c2, c3 = st.columns(3)
    if c1.button("Metal Rod"):
        st.session_state.alpha, st.session_state.L = 0.5, 1.0
    if c2.button("Ceramic"):
        st.session_state.alpha, st.session_state.L = 0.01, 2.0
    if c3.button("Exam Q3"):
        st.session_state.alpha, st.session_state.L, st.session_state.T_final = 0.1, 1.0, 0.5

    alpha = st.slider("Diffusivity α", 0.001, 1.0, st.session_state.get('alpha', 0.1), 0.001, help="Material property in u_t = α u_xx")
    L = st.number_input("Domain Length L [m]", 0.1, 10.0, st.session_state.get('L', 1.0))
    Nx = st.slider("Grid Points Nx", 50, 500, 200, help="Higher = more accurate, slower")
    T_final = st.slider("Final Time [s]", 0.1, 10.0, st.session_state.get('T_final', 1.0))
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
    
    st.info(f"**Stability:** r = {r:.3f} ✅  \n**Steps:** {Nt}  \n**dt:** {dt:.5f}")

# 3. MAIN AREA: PLOT + RESULTS
tab1, tab2, tab3 = st.tabs(["🔥 Heat Equation", "🎯 Root Finder 🔒", "📈 ODE Solver 🔒"])

with tab1:
    col_plot, col_data = st.columns([3, 1])

    if st.button("▶️ Solve & Animate", type="primary", use_container_width=True):
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

        # ANIMATION
        with col_plot:
            st.subheader("Temperature Diffusion")
            plot_spot = st.empty()
            for i, frame in enumerate(frames):
                fig, ax = plt.subplots(figsize=(9,4), facecolor="#0E1117")
                ax.plot(x, frame, color="#FF4B4B", linewidth=2.5)
                if uploaded_file: ax.plot(df_user['x'], df_user['u_final'], 'b--', label="Your Data")
                ax.set_facecolor("#1A1C23")
                ax.tick_params(colors='white')
                ax.set_ylim(-0.1, 1.1)
                ax.set_xlabel("x [m]", color="white")
                ax.set_ylabel("u(x,t)", color="white")
                ax.set_title(f"Time: {i*dt*(Nt//100):.3f} s", color="white")
                ax.legend()
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
            
            st.warning("**Pro Feature**: Export PDF Report")
            st.link_button("Upgrade to Pro $9/mo", "https://buy.stripe.com/test_xxxx", use_container_width=True)

with tab2:
    st.subheader("🎯 Root Finder")
    st.write("Newton-Raphson, Secant, Bisection with convergence plots.")
    st.button("🔒 Unlock in Pro Plan $19/mo", disabled=True)

with tab3:
    st.subheader("📈 ODE Solver")
    st.write("RK4, Adaptive Euler with error estimation.")
    st.button("🔒 Unlock in Pro Plan $19/mo", disabled=True)            st.success(f"Solved!")
            st.metric("Max Temp Start", f"{max(frames[0]):.3f}")
            st.metric("Max Temp End", f"{max(frames[-1]):.3f}")

            df = pd.DataFrame({"x": x, "u_final": frames[-1]})
            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "solution.csv", "text/csv")

            # PDF REPORT UPSELL
            st.divider()
            st.warning("**Pro Feature**: Export PDF Report with plots + parameters")
            if st.button("Unlock Pro $9/mo"):
                st.link_button("Go to Checkout", "https://buy.stripe.com/test_xxxx") # REPLACE THIS
