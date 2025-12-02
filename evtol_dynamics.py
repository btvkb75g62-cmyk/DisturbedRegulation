import numpy as np
from RLQRD import RLQRD, MRLQRD
from scipy import linalg
import pandas as pd

def generate_evtol_matrices(
    Ts,          # Sample time
    m_bar,       # Nominal mass (bar_m)
    Iyy,         # Moment of inertia about Y-axis (I_yy)
    Xu,          # Aerodynamic derivative X_u (from linearized dynamics, typically negative)
    g,           # Acceleration due to gravity
    u0,          # Equilibrium velocity u (for linearization, assumed 0 for hovering)
    V_max,       # Maximum gust magnitude
    delta_m,    # Mean radius of influence of the gust region (m)
):



    # --- Nominal Matrices ---
    F_Long = np.array([
        [Xu * Ts / m_bar + 1, 0, -g * Ts, 0],
        [0, 1, 0, Ts * u0],
        [0, 0, 1, 1],
        [0, 0, 0, 1]
    ])

    G_Long = np.array([
        [Ts / m_bar, 0, 0],
        [0, Ts / m_bar, 0],
        [0, 0, 0],
        [0, 0, Ts / Iyy]
    ])

    # W_Long represents the effect of gust disturbances [w_x, w_z]' on the states
    # based on the linearized equations where disturbances w_x/m and w_z/m appear.
    W_Long = np.array([
        [Ts * V_max / (2 * m_bar), 0],
        [0, Ts * V_max / (2 * m_bar)],
        [0, 0],
        [0, 0]
    ])

    # --- Uncertainty Components ---
    # M_k matrix (4x4)
    # As per the provided equations, M_k is a 4x4 block matrix.
    M_k = np.block([
        [np.eye(2), np.eye(2)],
        [np.zeros((2, 2)), np.zeros((2, 2))]
    ])

    # E_F matrix (4x4)
    # This matrix captures how the mass uncertainty affects the F matrix terms.
    # The term -X_u*T_s/m_bar^2 comes from the Taylor expansion of 1/m.
    E_F = np.array([
        [-Xu * Ts * delta_m / (m_bar**2), 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    # E_G matrix (4x3)
    # This matrix captures how the mass uncertainty affects the G matrix terms.
    # Note: The provided LaTeX for E_G shows a 4x4 matrix, but G_Long is 4x3.
    # We assume the last column of the provided E_G should be omitted or zero
    # to match the input dimension 'm=3'.
    E_G = np.array([
        [-Ts * delta_m / (m_bar**2), 0, 0],
        [0, -Ts * delta_m / (m_bar**2), 0],
        [0, 0, 0],
        [0, 0, 0]
    ])

    # E_W matrix (4x2)
    # This matrix captures how gust-related uncertainty (implicitly through V_max/m^2 terms)
    # affects the W matrix.
    E_W = np.array([
        [-Ts * V_max * delta_m / (2 * m_bar**2), 0],
        [0, -Ts * V_max * delta_m / (2 * m_bar**2)],
        [- Ts * V_max / (2 * m_bar), 0],
        [0, - Ts * V_max / (2 * m_bar)]
    ])

    return F_Long, E_F, G_Long, E_G, W_Long, E_W, M_k

def check_robustness_condition(E_F_k, E_G_k, E_W_k, tol=None):
    """
    Checks the sufficient condition for the existence of control gains K_k and f_k
    that satisfy the robust control law equations:
    1) E_F_k + E_G_k K_k = 0
    2) E_W_k + E_G_k f_k = 0

    The condition is:
    rank([E_F_k | E_G_k]) = rank([E_G_k | E_W_k]) = rank(E_G_k)

    This means that E_F_k and E_W_k must lie within the range space of E_G_k.

    Parameters:
    - E_F_k (numpy.ndarray): The E_F matrix (l x n).
    - E_G_k (numpy.ndarray): The E_G matrix (l x m).
    - E_W_k (numpy.ndarray): The E_W matrix (l x p_w, where p_w is dimension of disturbance w_k).
    - tol (float, optional): Tolerance for rank computation. If None, numpy's
                             default tolerance based on machine epsilon is used.

    Returns:
    - bool: True if the robustness condition is met, False otherwise.
    """
    # Ensure all input matrices have the same number of rows 'l'
    if not (E_F_k.shape[0] == E_G_k.shape[0] == E_W_k.shape[0]):
        raise ValueError("All input matrices (E_F_k, E_G_k, E_W_k) must have the same number of rows.")

    # 1. Compute rank(E_G_k)
    rank_EG_k = np.linalg.matrix_rank(E_G_k, tol=tol)

    # 2. Form concatenated matrix [E_F_k | E_G_k] and compute its rank
    # Note: np.concatenate stacks arrays along a specified axis. axis=1 for horizontal stacking.
    concat_EF_EG_k = np.concatenate((E_F_k, E_G_k), axis=1)
    rank_EF_EG_k = np.linalg.matrix_rank(concat_EF_EG_k, tol=tol)

    # 3. Form concatenated matrix [E_G_k | E_W_k] and compute its rank
    concat_EG_EW_k = np.concatenate((E_G_k, E_W_k), axis=1)
    rank_EG_EW_k = np.linalg.matrix_rank(concat_EG_EW_k, tol=tol)

    # Print ranks for analysis
    print(f"Rank(E_G_k): {rank_EG_k}")
    print(f"Rank([E_F_k | E_G_k]): {rank_EF_EG_k}")
    print(f"Rank([E_G_k | E_W_k]): {rank_EG_EW_k}")

    # Check the condition: all three ranks must be equal
    condition_met = (rank_EF_EG_k == rank_EG_k) and \
                    (rank_EG_EW_k == rank_EG_k)

    if condition_met:
        print("\nRobustness condition is MET: E_F_k and E_W_k lie within the range space of E_G_k.")
    else:
        print("\nRobustness condition is NOT MET: E_F_k or E_W_k (or both) do not lie within the range space of E_G_k.")

    return condition_met

def read_longitudinal_matrices(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    matrices = {}
    current_key = None
    buffer = []

    for line in lines:
        line = line.strip()
        
        # Identify matrix headers
        if line.startswith('---'):
            continue  # Skip section headers
        elif line.endswith(':'):
            if current_key and buffer:
                # Save the previous matrix
                matrices[current_key] = np.array(eval(''.join(buffer)))
                buffer = []
            current_key = line[:-1].strip()
        elif line.startswith('[') or line.endswith(']'):
            buffer.append(line)
    
    # Save the last matrix
    if current_key and buffer:
        matrices[current_key] = np.array(eval(''.join(buffer)))

    return matrices

def simulate_eVTOL(K_mrlqrd, K_rlqrd, K_rlqr, K_Hinf, K_ellip, K_lqr, Mark_Lambda, F_Long, G_Long, W_Long, E_F, E_G, E_W, M_k, Ts, x0):

    # Simulation parameters
    dt = Ts  # Time step
    t_end = 120.0  # End time
    t = np.arange(0, t_end, dt)

    # State initialization
    x = x0
    xlqr = x0
    xrlqr = x0
    xHinf = x0
    xellip = x0

    phi = np.ones((2, 1))  # Initial disturbance state

    Lambda = Mark_Lambda[0]  # Initial Markov state

    # Storage for results
    states = []
    xlqr_states = []
    xrlqr_states = []
    xHinf_states = []
    xellip_states = []

    inputs = []
    xlqr_inputs = []
    xrlqr_inputs = []
    xHinf_inputs = []
    xellip_inputs = []

    disturbances = []

    u = np.zeros((G_Long.shape[1], 1))  # Initial control input
    ulqr = np.zeros((K_lqr.shape[0], 1))  # Initial LQR control input
    urlqr = np.zeros((K_rlqr.shape[0], 1))  # Initial RLQR control input
    uHinf = np.zeros((K_Hinf.shape[0], 1))  # Initial H∞ control input
    uellip = np.zeros((K_ellip.shape[0], 1))  # Initial Ellipsoid control input

    for k in t:

        if k < t_end/2:
            i = 0
        if k > 0.5*t_end and k < 100:
            i = 1
        if k > 100:
            phi = np.ones((2, 1))
            i = 0

        Lambda = Mark_Lambda[i]

        #u = K_rlqrd @ np.vstack([x, phi])
        u = K_mrlqrd[i] @ np.vstack([x, phi])

        urlqr = K_rlqr[:, :F_Long.shape[1]] @ xrlqr

        ulqr = K_lqr @ xlqr

        uHinf = K_Hinf @ xHinf

        uellip = K_ellip @ xellip

        Delta = linalg.block_diag(-1 * np.eye(2),  np.cos(k) * np.eye(2))

        frequency = 5 * np.pi / 10  

        Delta_w = np.block([[-1 * np.eye(2),    np.zeros((2, 2))], 
                            [-1 * np.cos(frequency*k) * np.eye(2), np.cos(frequency*k) * np.eye(2)]])

        D_F = M_k @ Delta @ E_F
        D_G = M_k @ Delta @ E_G
        D_W = M_k @ Delta_w @ E_W

        states.append(x.copy())
        xlqr_states.append(xlqr.copy())
        xrlqr_states.append(xrlqr.copy())
        xHinf_states.append(xHinf.copy())
        xellip_states.append(xellip.copy())

        inputs.append(u.copy())
        xlqr_inputs.append(ulqr.copy())
        xrlqr_inputs.append(urlqr.copy())
        xHinf_inputs.append(uHinf.copy())
        xellip_inputs.append(uellip.copy())

        disturbances.append(((W_Long + D_W) @ phi).copy())

        x =      (F_Long + D_F) @ x       +  (G_Long + D_G) @ u      +   (W_Long + D_W) @ phi
        xlqr =   (F_Long + D_F) @ xlqr    +  (G_Long + D_G) @ ulqr   +   (W_Long + D_W) @ phi
        xrlqr =  (F_Long + D_F) @ xrlqr   +  (G_Long + D_G) @ urlqr  +   (W_Long + D_W) @ phi
        xHinf =  (F_Long + D_F) @ xHinf   +  (G_Long + D_G) @ uHinf  +   (W_Long + D_W) @ phi
        xellip = (F_Long + D_F) @ xellip  +  (G_Long + D_G) @ uellip +   (W_Long + D_W) @ phi

        phi = Lambda @ phi

    return np.array(states), np.array(inputs), np.array(disturbances), t, np.array(xlqr_states), np.array(xlqr_inputs), np.array(xrlqr_states), np.array(xrlqr_inputs), np.array(xHinf_states), np.array(xHinf_inputs), np.array(xellip_states), np.array(xellip_inputs)

# --- Example Usage ---
# Define parameters (using reasonable placeholder values where not explicitly given in text)
rho = 1.225  # Air density at sea level, kg/m^3
S_x = 3 # Reference area in the x-direction, m^2 (placeholder, actual value needed from vehicle specs)
Cdx = 0.74  # Drag coefficient (placeholder, typical value for eVTOLs)
Ts_val = 0.01  # Sample time in seconds
m_bar_val = 500.0  # Nominal mass in kg
Iyy_val = 732.0 # Moment of inertia, kg*m^2 (placeholder, actual value needed from vehicle specs)
Xu_val = -rho*S_x*Cdx   # Aerodynamic derivative Xu (placeholder, typically negative for drag)
g_val = 9.81   # Acceleration due to gravity, m/s^2
u0_val = 5.0   # Equilibrium forward velocity, m/s (0 for hovering)
V_max_val = 2.0 # Maximum gust magnitude, m/s
dm_val = 10.0  # Mean radius of influence of gust region, meters
V_inf_val = 1.0 # Aircraft velocity for gust penetration, m/s (even in hover, small movement can encounter gust)
delta_m = 34.0  # Nominal mass uncertainty, kg (as per the example, can be adjusted)

# Example time step and a specific value for the normalized mass uncertainty Delta^A
k_step = 10    # Time step k
delta_A_val = 0.5 # Example: Normalized mass uncertainty, ranging from -1 to 1.
                  # A value of 0.5 means the actual mass deviation is 0.5 * 34 kg.

# Generate the matrices
F, dF, G, dG, W, dW, M = generate_evtol_matrices(
    Ts_val, m_bar_val, Iyy_val, Xu_val, g_val, u0_val,
    V_max_val, delta_m
)

# Export matrices to a text file
output_file = "evtol_longitudinal_matrices.txt"
with open(output_file, 'w') as f:
    f.write("--- Longitudinal Dynamics Matrices ---\n")
    f.write(f"Nominal F_Long:\n{F}\n\n")
    f.write(f"Uncertainty E_F:\n{dF}\n\n")
    f.write(f"Nominal G_Long:\n{G}\n\n")
    f.write(f"Uncertainty E_G:\n{dG}\n\n")
    f.write(f"Nominal W_Long:\n{W}\n\n")
    f.write(f"Uncertainty E_W:\n{dW}\n\n")

# System dimensions
n = F.shape[0]           # state dimension
m = G.shape[1]           # input dimension
w_dim = W.shape[1]       # dimension of disturbance

# Cost matrices
Q = np.eye(n)
R = np.eye(m)

decay = 0.9992  # Decay factor for disturbance state
Lambda = np.array([
    [decay, 0.1],
    [0, decay]
])  

Lambda2 = np.array([
    [0.95, 0],
    [0, 0.95]
])  

# Make a tensor of matrices
Lambda_Mark = np.stack([Lambda, Lambda2], axis=0)
Probability = np.array([[0.2, 0.8], 
                        [0.8, 0.2]])  # Probability distribution for Markovian states

## Standard LQR controller
P_standard = linalg.solve_discrete_are(F, G, Q[:n, :n], R)
K_lqr = - np.linalg.inv(R + G.T @ P_standard @ G) @ (G.T @ P_standard @ F)

# Check robustness condition
E_F = dF
E_G = dG
E_W = dW
check_robustness_condition(E_F, E_G, E_W)

# Instantiate controller
# Robust Controller with Disturbances
ROC = RLQRD(F, G, W, dF, dG, dW, M, Lambda, Q, R, mu=1e12, beta=1.10)
L_k, K_rlqrd, P_k = ROC.main(solution_type='auto')

print("Robust Controller with Disturbances")
print(f"K_k: {K_rlqrd}")
print(f"P_k: {P_k}")

# Robust Controller without Disturbances
ROC_no_dist = RLQRD(F, G, 0*W, dF, dG, 0*dW, M, Lambda, Q, R, mu=1e12, beta=1.10)
L_k, K_rlqr, P_k = ROC_no_dist.main(solution_type='auto')

print("Robust Controller without Disturbances")
print(f"K_k: {K_rlqr}")
print(f"P_k: {P_k}")

# Markovian Robust Controller with Disturbances
MROC = MRLQRD(F, G, W, dF, dG, dW, M, Lambda_Mark, Q, R, Prob=Probability, mu=1e12, beta=1.10)
L_k, K_mrlqrd, P_k = MROC.main(solution_type='auto')

print("Markovian Robust Controller with Disturbances")
print(f"K_k: {K_mrlqrd}")
print(f"P_k: {P_k}")

## H infinity controller

K_Hinf =  np.array([[-1.11488573e-03, -7.54466920e-05,  8.37602125e-04,  1.81949771e-03],
                [-5.68329115e-05, -4.45910132e-03, 2.42460176e-04, -2.97451366e-05],
                [ 9.01558543e-04,  2.11918530e-05, -1.89327955e-03, -8.31782162e-03]])

## Attractive Ellipsoid controller

K_ellip =  np.array([[-4.98312227e+04,  6.21503394e+00,  4.84749390e+03, -6.63940044e+01],
                      [ 2.24932298e-01, -4.98421445e+04,  1.91816832e+00, -2.48975661e+03],
                      [ 3.04615926e+01,  2.59236850e+00, -5.70775410e+04, -1.30274645e+05]])

K_ellip = np.array([[-4.98548820e+04,  1.35011496e+00,  5.59260478e+03,  5.69850225e+02],
                    [-1.13766154e+01, -4.99507110e+04,  1.80349143e+02, -2.28054065e+03],
                    [ 3.86066997e+01,  5.61099219e-01, -5.04407418e+04, -1.24226827e+05]])

# Simulate the eVTOL dynamics
x0 = np.array([[0.0], [0.0], [0.0], [0.0]])  # Initial state (position and velocity)
states, inputs, disturbances, time, xlqr_states, xlqr_inputs, xrlqr_states, xrlqr_inputs, xHinf_states, xHinf_inputs, xellip_states, xellip_inputs = simulate_eVTOL(K_mrlqrd ,K_rlqrd, K_rlqr, K_Hinf, K_ellip, K_lqr, Lambda_Mark, F, G, W, dF, dG, dW, M, Ts_val, x0)

# Prepare results for CSV
results_dict = {
    'time': time.flatten() if time.ndim > 1 else time,
    'RLQRD_state_norm': np.linalg.norm(states[:, :n], axis=1).flatten(),
    'LQR_state_norm': np.linalg.norm(xlqr_states[:, :n], axis=1).flatten(),
    'RLQR_state_norm': np.linalg.norm(xrlqr_states[:, :n], axis=1).flatten(),
    'Hinf_state_norm': np.linalg.norm(xHinf_states[:, :n], axis=1).flatten(),
    'AEM_state_norm': np.linalg.norm(xellip_states[:, :n], axis=1).flatten(),
    'RLQRD_input_norm': np.linalg.norm(inputs, axis=1).flatten(),
    'LQR_input_norm': np.linalg.norm(xlqr_inputs, axis=1).flatten(),
    'RLQR_input_norm': np.linalg.norm(xrlqr_inputs, axis=1).flatten(),
    'Hinf_input_norm': np.linalg.norm(xHinf_inputs, axis=1).flatten(),
    'AEM_input_norm': np.linalg.norm(xellip_inputs, axis=1).flatten(),
    'disturbance_norm': np.linalg.norm(disturbances, axis=1).flatten()
}

df_results = pd.DataFrame(results_dict)
df_results.to_csv("evtol_simulation_results_massminus.csv", index=False)
print("Results saved to evtol_simulation_results.csv")


# Print simulation results
print("Simulation completed.")

# Plotting results (optional)
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))

# === Subplot 1 ===
plt.subplot(3, 1, 1)
plt.plot(time, np.linalg.norm(states[:, :n], axis=1), 'b', label='RLQRD')
plt.plot(time, np.linalg.norm(xlqr_states[:, :n], axis=1), 'k:', label='LQR')
plt.plot(time, np.linalg.norm(xrlqr_states[:, :n], axis=1), 'r--', label='RLQR')
plt.plot(time, np.linalg.norm(xHinf_states[:, :n], axis=1), 'g-.', label=r'$H_{\infty}$')
plt.plot(time, np.linalg.norm(xellip_states[:, :n], axis=1), 'm--', label='AEM')
plt.ylim([0, 0.3])
plt.title('States over Time', fontsize=18)
#plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('States', fontsize=14)
plt.grid()
plt.legend(fontsize=16)
plt.tick_params(axis='both', labelsize=16)

# === Subplot 2 ===
plt.subplot(3, 1, 2)
plt.plot(time, np.linalg.norm(inputs, axis=1), 'b', label='RLQRD')
plt.plot(time, np.linalg.norm(xlqr_inputs, axis=1), 'k:', label='LQR')
plt.plot(time, np.linalg.norm(xrlqr_inputs, axis=1), 'r--', label='RLQR')
plt.plot(time, np.linalg.norm(xHinf_inputs, axis=1), 'g-.', label=r'$H_{\infty}$')
plt.plot(time, np.linalg.norm(xellip_inputs, axis=1), 'm--', label='AEM')
plt.title('Control Inputs over Time', fontsize=18)
#plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Inputs', fontsize=14)
plt.grid()
plt.legend(fontsize=16)
plt.tick_params(axis='both', labelsize=16)

# === Subplot 3 ===
plt.subplot(3, 1, 3)
plt.plot(time, np.linalg.norm(disturbances, axis=1), 'k', label='Disturbances Norm')
plt.title('Disturbances over Time', fontsize=18)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Disturbances', fontsize=14)
plt.grid()
plt.legend(fontsize=16)
plt.tick_params(axis='both', labelsize=16)

plt.tight_layout()

from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


fig = plt.figure(figsize=(20, 12))
ax = fig.add_subplot(111, projection='3d')

# Data series
state_series = [
    np.linalg.norm(xellip_states[:, :n], axis=1),
    np.linalg.norm(states[:, :n], axis=1),
    np.linalg.norm(xrlqr_states[:, :n], axis=1),
    np.linalg.norm(xlqr_states[:, :n], axis=1),
    np.linalg.norm(xHinf_states[:, :n], axis=1),
]

state_MSE = [
    np.mean(np.square(xellip_states[:, :n])),
    np.mean(np.square(states[:, :n])),
    np.mean(np.square(xrlqr_states[:, :n])),
    np.mean(np.square(xlqr_states[:, :n])),
    np.mean(np.square(xHinf_states[:, :n])),
]

print("State MSEs: \n")
print(f"AEM: {state_MSE[0]:.8f}, RLQRD: {state_MSE[1]:.4f}, RLQR: {state_MSE[2]:.4f}, LQR: {state_MSE[3]:.4f}, H∞: {state_MSE[4]:.4f}")

# Depth levels and labels
depths = [0, 1, 2, 3, 4]
labels = ['AEM', 'RLQRD', 'RLQR', 'LQR', r'$H_{\infty}$']
colors = ['m', 'b', 'k', 'r', 'g']
linestyles = ['-', '-', '-', '-', '-']

# Plot lines with depth
for depth, series, label, color, ls in zip(depths, state_series, labels, colors, linestyles):
    ax.plot(time, series, zs=depth, zdir='y', label=label, color=color, linestyle=ls, linewidth=2)

# Label settings
ax.set_xlabel('Time (s)', labelpad=12)
ax.set_zlabel(r'$||x||_{2}$', labelpad=12)
ax.set_zlim(0, 5)
ax.set_yticks(depths)
ax.set_yticklabels(labels, fontsize=10)
ax.set_title('State Norms over Time with Depth Plot', pad=20)

# Aspect ratio
ax.set_box_aspect([12, 6, 6])

# CLEANUP: remove background panes and grid
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('k')

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = True

ax.grid(False)
ax.tick_params(axis='both', which='major', labelsize=12)

ax.legend(loc='upper left', fontsize=12, frameon=False)
plt.tight_layout()
plt.show()