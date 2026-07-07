import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

st.set_page_config(page_title="Heat Equation Pro", layout="wide")
st.title("🔥 Heat Equation Pro v1.1")
st.caption("1D Heat Solver with Animation. Built with FDM + Stability Analysis.")

# SIDEBAR
st.sidebar.header("Parameters")
alpha = st.sidebar.slider("Diffusivity α", 0.01, 1.0, 0.1, 0.01)
L = st.sidebar.number_input("Domain Length L", 1.0, 10.0, 1.0)
Nx = st.sidebar.slider("Grid Points Nx", 50, 300, 100)
T_final = st.sidebar.slider("Final Time", 0.1, 5.0, 1.0)
bc_type = st.sidebar.selectbox("Boundary Conditions", ["Dirichlet: u=0", "Neumann: du/dx=0"])

dx = L / (Nx-1)
dt = 0.4 * dx**2 / alpha
Nt = int(T_final / dt)
r = alpha * dt / dx**2

st.sidebar.info(f"Auto dt={dt:.5f} for stability. r={r:.3f}. Total steps: {Nt}")

# TABS
tab1, tab2, tab3 = st.tabs(["🔥 Heat Equation", "🎯 Root Finder 🔒", "📈 ODE Solver 🔒"])

with tab1:
    col1, col2 = st.columns([3,1])

    if st.button("Solve & Animate PDE", type="primary"):
        x = np.linspace(0, L, Nx)
        u = np.sin(np.pi * x / L)
        frames = [u.copy()]

        progress = st.progress(0, "Solving...")

        # FDM LOOP
        for n in range(Nt):
            u_new = u.copy()
            u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])

            if bc_type.startswith("Dirichlet"):
                u_new[0] = 0; u_new[-1] = 0
            else:
                u_new[0] = u_new[1]; u_new[-1] = u_new[-2]
            u = u_new

            if n % max(1, Nt//100) == 0:
                frames.append(u.copy())
            progress.progress((n+1)/Nt)

        progress.empty()

        # ANIMATION PLOT
        with col1:
            st.subheader("Animation: Heat Diffusion")
            plot_placeholder = st.empty()

            for i, frame in enumerate(frames):
                fig, ax = plt.subplots(figsize=(8,4))
                ax.plot(x, frame, color='red')
                ax.set_ylim(-0.1, 1.1)
                ax.set_xlabel("x")
                ax.set_ylabel("u(x,t)")
                ax.set_title(f"Time: {i*dt* (Nt//100):.3f} s")
                ax.grid(True)
                plot_placeholder.pyplot(fig)
                plt.close(fig)

        # RESULTS + DOWNLOAD
        with col2:
            st.success(f"Solved!")
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

with tab2:
    st.subheader("🎯 Root Finder - Newton, Secant, Fixed Point")
    st.write("Solve f(x)=0 with convergence guarantees from Banach Fixed Point Theorem.")
    st.button("🔒 Unlock in Pro Plan $19/mo")

with tab3:
    st.subheader("📈 ODE Solver - RK4, Euler")
    st.write("Solve IVPs y'=f(t,y) with adaptive step size and error control.")
    st.button("🔒 Unlock in Pro Plan $19/mo", key="ode")
