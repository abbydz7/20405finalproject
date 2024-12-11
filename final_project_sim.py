
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the ODE system for the model
def model(y, t, params):
    AHL, OxyR, Act1, Rep1, Act2, Rep2, Act3, Act4, BDE = y
    (p_Act1, d_Act1, K_Act1, n_Act1, p_Act2, d_Act2, K_Act2, n_Act2,
     p_Rep1, d_Rep1, K_Rep1, n_Rep1, p_Act3, d_Act3, K_Act3, n_Act3,
     p_Act4, d_Act4, K_Act4, n_Act4, p_Rep2, d_Rep2, K_Rep2, n_Rep2,
     d_BDE, input_rate_AHL, frequency_AHL, input_rate_OxyR, frequency_OxyR) = params

    # Input signals
    AHL_input = input_rate_AHL * (np.sin(2 * np.pi * frequency_AHL * t) > 0)
    OxyR_input = input_rate_OxyR * (np.sin(2 * np.pi * frequency_OxyR * t) > 0)



    # Ensure non-negative values to avoid numerical errors
    AHL_safe = max(AHL, 1e-9)
    OxyR_safe = max(OxyR, 1e-9)
    Act1_safe = max(Act1, 1e-9)
    Rep1_safe = max(Rep1, 1e-9)
    Act2_safe = max(Act2, 1e-9)
    Rep2_safe = max(Rep2, 1e-9)
    Act3_safe = max(Act3, 1e-9)
    Act4_safe = max(Act4, 1e-9)

    dAHL_dt = AHL_input - 0.5 * AHL_safe
    dOxyR_dt = OxyR_input - 0.5 * OxyR_safe
    dAct1_dt = p_Act1 * (AHL_safe**n_Act1) / (K_Act1**n_Act1 + AHL_safe**n_Act1) - d_Act1 * Act1_safe
    dRep1_dt = p_Rep1 * (Act1_safe**n_Rep1) / (K_Rep1**n_Rep1 + Act1_safe**n_Rep1) - d_Rep1 * Rep1_safe
    dAct2_dt = p_Act2 * (OxyR_safe**n_Act2) / (K_Act2**n_Act2 + OxyR_safe**n_Act2) - d_Act2 * Act2_safe
    dRep2_dt = p_Rep2 * (Act2_safe**n_Rep2) / (K_Rep2**n_Rep2 + Act2_safe**n_Rep2) - d_Rep2 * Rep2_safe
    dAct3_dt = p_Act3 * (Act1_safe**n_Act3) / (K_Act3**n_Act3 + Act1_safe**n_Act3) * (1 / (1 + (Rep1_safe / K_Rep1)**n_Rep1)) - d_Act3 * Act3_safe
    dAct4_dt = p_Act4 * (Act2_safe**n_Act4) / (K_Act4**n_Act4 + Act2_safe**n_Act4) * (1 / (1 + (Rep2_safe / K_Rep2)**n_Rep2)) - d_Act4 * Act4_safe
    dBDE_dt = (Act3_safe**n_Act3) / (K_Act3**n_Act3 + Act3_safe**n_Act3) * (Act4_safe**n_Act4) / (K_Act4**n_Act4 + Act4_safe**n_Act4) - d_BDE * BDE

    return [dAHL_dt, dOxyR_dt, dAct1_dt, dRep1_dt, dAct2_dt, dRep2_dt, dAct3_dt, dAct4_dt, dBDE_dt]

# Function to compute BDE metrics
def compute_bde_metrics(time, bde_data):
    peak = np.max(bde_data)
    ttp = time[np.argmax(bde_data)]
    threshold = 0.5 * peak
    pulse_duration = np.sum(bde_data > threshold) * (time[1] - time[0])
    auc = np.trapz(bde_data, time)
    return {"peak": peak, "time_to_peak": ttp, "pulse_duration": pulse_duration, "auc": auc}

# Streamlit app
st.title("Integrated QS and OS IFFLs")
st.sidebar.header("Parameter Controls")

# Input parameters
st.sidebar.subheader("AHL and OxyR Inputs")
input_rate_AHL = st.sidebar.slider("Input Rate (AHL)", 0.1, 10.0, 5.0)
frequency_AHL = st.sidebar.slider("Frequency (AHL)", 0.001, 0.1, 0.01, step=0.001)
input_rate_OxyR = st.sidebar.slider("Input Rate (OxyR)", 0.1, 10.0, 5.0)
frequency_OxyR = st.sidebar.slider("Frequency (OxyR)", 0.001, 0.1, 0.01, step=0.001)

# Activator and Repressor parameters
st.sidebar.subheader("Activator 1")
p_Act1 = st.sidebar.slider("Production Rate (Act1)", 0.1, 10.0, 5.0)
d_Act1 = st.sidebar.slider("Degradation Rate (Act1)", 0.01, 1.0, 0.2)
K_Act1 = st.sidebar.slider("K (Act1)", 0.1, 10.0, 2.0)
n_Act1 = st.sidebar.slider("Hill Coefficient (Act1)", 1.0, 4.0, 2.0)

st.sidebar.subheader("Activator 2")
p_Act2 = st.sidebar.slider("Production Rate (Act2)", 0.1, 10.0, 5.0)
d_Act2 = st.sidebar.slider("Degradation Rate (Act2)", 0.01, 1.0, 0.2)
K_Act2 = st.sidebar.slider("K (Act2)", 0.1, 10.0, 2.0)
n_Act2 = st.sidebar.slider("Hill Coefficient (Act2)", 1.0, 4.0, 2.0)

st.sidebar.subheader("Repressor 1")
p_Rep1 = st.sidebar.slider("Production Rate (Rep1)", 0.1, 10.0, 1.0)
d_Rep1 = st.sidebar.slider("Degradation Rate (Rep1)", 0.01, 1.0, 0.5)
K_Rep1 = st.sidebar.slider("K (Rep1)", 0.1, 10.0, 1.0)
n_Rep1 = st.sidebar.slider("Hill Coefficient (Rep1)", 1.0, 4.0, 3.0)

st.sidebar.subheader("Activator 3")
p_Act3 = st.sidebar.slider("Production Rate (Act3)", 0.1, 10.0, 5.0)
d_Act3 = st.sidebar.slider("Degradation Rate (Act3)", 0.01, 1.0, 0.2)
K_Act3 = st.sidebar.slider("K (Act3)", 0.1, 10.0, 2.0)
n_Act3 = st.sidebar.slider("Hill Coefficient (Act3)", 1.0, 4.0, 2.0)

st.sidebar.subheader("Activator 4")
p_Act4 = st.sidebar.slider("Production Rate (Act4)", 0.1, 10.0, 5.0)
d_Act4 = st.sidebar.slider("Degradation Rate (Act4)", 0.01, 1.0, 0.2)
K_Act4 = st.sidebar.slider("K (Act4)", 0.1, 10.0, 2.0)
n_Act4 = st.sidebar.slider("Hill Coefficient (Act4)", 1.0, 4.0, 2.0)

st.sidebar.subheader("Repressor 2")
p_Rep2 = st.sidebar.slider("Production Rate (Rep2)", 0.1, 10.0, 1.0)
d_Rep2 = st.sidebar.slider("Degradation Rate (Rep2)", 0.01, 1.0, 0.5)
K_Rep2 = st.sidebar.slider("K (Rep2)", 0.1, 10.0, 1.0)
n_Rep2 = st.sidebar.slider("Hill Coefficient (Rep2)", 1.0, 4.0, 3.0)

# BDE Parameters
st.sidebar.subheader("BDE Parameters")
d_BDE = st.sidebar.slider("Degradation Rate (BDE)", 0.01, 1.0, 0.1)

# Simulation time
st.sidebar.subheader("Simulation Time")
T = st.sidebar.slider("Simulation Time (hours)", 10, 1000, 100)
time = np.linspace(0, T, 5000)
y0 = [0, 0, 0, 0, 0, 0, 0, 0, 0]

# Model parameters
params = [p_Act1, d_Act1, K_Act1, n_Act1,  # Act1
          p_Act2, d_Act2, K_Act2, n_Act2,  # Act2
          p_Rep1, d_Rep1, K_Rep1, n_Rep1,  # Repressor 1
          p_Act3, d_Act3, K_Act3, n_Act3,  # Act3
          p_Act4, d_Act4, K_Act4, n_Act4,  # Act4
          p_Rep2, d_Rep2, K_Rep2, n_Rep2,  # Repressor 2
          d_BDE, input_rate_AHL, frequency_AHL, input_rate_OxyR, frequency_OxyR]

# Solve ODEs
results = odeint(model, y0, time, args=(params,))
AHL, OxyR, Act1, Rep1, Act2, Rep2, Act3, Act4, BDE = results.T

# Compute BDE metrics
bde_metrics = compute_bde_metrics(time, BDE)

# Display metrics
st.header("BDE Metrics Summary")
st.write(f"**Peak BDE Concentration:** {bde_metrics['peak']:.2f}")
st.write(f"**Time to Peak BDE:** {bde_metrics['time_to_peak']:.2f} hours")
st.write(f"**Pulse Duration (BDE > 50% peak):** {bde_metrics['pulse_duration']:.2f} hours")
st.write(f"**Total Enzyme Production (AUC):** {bde_metrics['auc']:.2f}")

# Plots
st.header("Plots")

# Plot 1: Quorum Sensing IFFL Dynamics
fig1, ax1 = plt.subplots()
ax1.plot(time, AHL, label="AHL")
ax1.plot(time, Act1, label="Activator 1")
ax1.plot(time, Rep1, label="Repressor 1")
ax1.plot(time, Act3, label="Activator 3")
ax1.legend()
ax1.set_title("Quorum Sensing IFFL Dynamics")
ax1.set_xlabel("Time (hours)")
ax1.set_ylabel("Concentration")
st.pyplot(fig1)

# Plot 2: Oxidative Stress IFFL Dynamics
fig2, ax2 = plt.subplots()
ax2.plot(time, OxyR, label="OxyR")
ax2.plot(time, Act2, label="Activator 2")
ax2.plot(time, Rep2, label="Repressor 2")
ax2.plot(time, Act4, label="Activator 4")
ax2.legend()
ax2.set_title("Oxidative Stress IFFL Dynamics")
ax2.set_xlabel("Time (hours)")
ax2.set_ylabel("Concentration")
st.pyplot(fig2)

# Plot 3: Final Output Dynamics
fig3, ax3 = plt.subplots()
ax3.plot(time, AHL, label="AHL")
ax3.plot(time, OxyR, label="OxyR")
ax3.plot(time, Act3, label="Activator 3")
ax3.plot(time, Act4, label="Activator 4")
ax3.plot(time, BDE, label="BDE", linestyle="--")
ax3.legend()
ax3.set_title("Final Output Dynamics")
ax3.set_xlabel("Time (hours)")
ax3.set_ylabel("Concentration")
st.pyplot(fig3)