import numpy as np
import scipy.linalg as linalg

class RobustOptimalController:
    def __init__(self, Q_seq, R_seq, alpha_seq):
        """
        Initialize the controller with sequences of Q, R, and alpha matrices.
        Each entry is assumed to correspond to timestep k.
        """
        self.Q_seq = Q_seq
        self.R_seq = R_seq
        self.alpha_seq = alpha_seq

    def compute_matrices(self, Fk, Gk, Wk, Ek_F, Ek_G, Ek_W, Mk, alpha_k):
        """Compute augmented matrices used in the controller."""
        n = Fk.shape[0]
        l = Wk.shape[1]

        F_aug = np.block([
            [Fk, np.zeros((n, n)), Wk],
            [np.eye(n), np.eye(n), np.zeros((n, l))],
            [np.zeros((1, n)), np.zeros((1, n)), alpha_k]
        ])
        G_aug = np.vstack([Gk, np.zeros((n, Gk.shape[1])), np.zeros((1, Gk.shape[1]))])

        E_F_aug = np.hstack([Ek_F, np.zeros((Ek_F.shape[0], n)), Ek_W])
        E_G_aug = Ek_G
        M_aug = np.vstack([Mk, np.zeros((1, Mk.shape[1])), np.zeros((n, Mk.shape[1]))])

        # Define bar_F, bar_G, and bar_I
        bar_I = np.vstack([np.eye(n + n + 1), np.zeros((E_F_aug.shape[0], n + n + 1))])
        bar_F = np.vstack([F_aug, E_F_aug])
        bar_G = np.vstack([G_aug, E_G_aug])

        return bar_I, bar_F, bar_G

    def step(self, Fk, Gk, Wk, Ek_F, Ek_G, Ek_W, Mk, P_k, Qk, Rk, alpha_k):

        bar_I, bar_F, bar_G = self.compute_matrices(Fk, Gk, Wk, Ek_F, Ek_G, Ek_W, Mk, alpha_k)

        inv_P_k = np.linalg.inv(P_k)

        temp = bar_I @ inv_P_k @ bar_I.T + bar_G @ np.linalg.inv(Rk) @ bar_G.T
        temp_inv = np.linalg.inv(temp)

        Lk = inv_P_k @ bar_I.T @ temp_inv @ bar_F 
        Kk = -np.linalg.inv(Rk) @ bar_G.T @ temp_inv @ bar_F 

        P_k = Qk + bar_F.T @ temp_inv @ bar_F

        return Lk, Kk, P_k


def simulate_system(v0, alpha, Fk, Gk, Wk, Ek_F, Ek_G, Ek_W, Mk, K_k, T, tol=1e-6):
    """
    Simulate the uncertain system with robust optimal control.

    Parameters:
    - Fk, Gk, Wk: Nominal matrices
    - Ek_F, Ek_G, Ek_W: Uncertainty effect matrices
    - Mk: Uncertainty multiplier
    - controller: An instance of RobustOptimalController
    - v0: Initial augmented state vector [x0; phi0]
    - T: Time horizon
    - tol: small tolerance to regularize matrix inversions

    Returns:
    - v_hist: History of states
    - u_hist: History of control inputs
    """

    n = Fk.shape[0]
    l = Wk.shape[1]
    
    v_k = v0.copy()
    s = v0[:n]  
    phi_k = v0[-1:]  # initial disturbance

    v_hist = [v_k.flatten()]
    u_hist = []

    for k in range(T):

        # Step 2: Compute control input
        u_k = K_k @ v_k
        u_hist.append(u_k.flatten())

        # Step 3: Sample an uncertainty Delta_k (bounded)
        d = Mk.shape[1] if Mk.ndim > 1 else 1  # disturbance dimension
        Delta_k = np.random.uniform(-1, 1, (d, d))  # simple uncertainty (could be more sophisticated)

        # Step 4: Compute uncertain matrices
        delta_Fk = Mk @ Delta_k @ Ek_F
        delta_Gk = Mk @ Delta_k @ Ek_G
        delta_Wk = Mk @ Delta_k @ Ek_W

        # Step 5: Update system state
        x_k = v_k[:n]

        s = s + x_k
        x_k1 = (Fk + delta_Fk) @ x_k + (Gk + delta_Gk) @ u_k + (Wk + delta_Wk) * phi_k
        phi_k = alpha*phi_k

        v_k = np.vstack((x_k1, s, 1 ))
        v_hist.append(v_k.flatten())

    v_hist = np.array(v_hist)
    u_hist = np.array(u_hist)
    return v_hist, u_hist

def simulate_multiple_systems(V0, Fk, Gk, Wk, Ek_F, Ek_G, Ek_W, Mk, K_k, T, tol=1e-6):
    """
    Simulate N uncertain systems starting from N different initial conditions.

    Parameters:
    - V0: Initial conditions matrix of shape (N, n+l)
    - Other parameters: as before

    Returns:
    - V_hist: array of shape (N, T+1, n+l) with the evolution of each initial condition
    - U_hist: array of shape (N, T, m) with control inputs applied
    """
    n = Fk.shape[0]
    m = Gk.shape[1]
    l = Wk.shape[1]

    N = V0.shape[0]

    V_hist = np.zeros((N, T+1, n+l))
    U_hist = np.zeros((N, T, m))

    for i in range(N):
        v_k = V0[i].reshape(-1, 1)  # (n+l, 1)

        V_hist[i, 0, :] = v_k.flatten()

        for k in range(T):
            # Step 2: Control input
            u_k = K_k @ v_k
            U_hist[i, k, :] = u_k.flatten()

            # Step 3: Uncertainty
            d = Mk.shape[1] if Mk.ndim > 1 else 1
            Delta_k = np.random.uniform(-1, 1, (d, d))

            # Step 4: Apply uncertain dynamics
            delta_Fk = Mk @ Delta_k @ Ek_F
            delta_Gk = Mk @ Delta_k @ Ek_G
            delta_Wk = Mk @ Delta_k @ Ek_W

            x_k = v_k[:n]
            phi_k = v_k[n:]
            s = s + x_k

            x_k1 = (Fk + delta_Fk) @ x_k + (Gk + delta_Gk) @ u_k + (Wk + delta_Wk) @ phi_k

            v_k = np.vstack((x_k1, 1))
            V_hist[i, k+1, :] = v_k.flatten()

    return V_hist, U_hist


def main():

    # Example matrices for timestep k
    Fk = np.array([[1.1, 0, 0], 
                   [0, 0, 1.2],
                   [-1, 1, 0]])  # state transition matrix
    Gk = np.array([[0, 1],
                   [1, 1],
                   [-1, 0]])  # control input matrix
    Wk = np.array([[0.5],
                   [0],
                   [0]])  # disturbance vector
    alpha = 1
    
    Ek_F = np.array([[0.4, 0.5, -0.6]])
    Ek_G = np.array([[0.4, -0.4]])
    Ek_W = np.array([[0.1]])

    Mk = np.array([[0.7],
                   [0.5],
                   [-0.7]])  # disturbance input matrix

    # Dimensions
    n = Fk.shape[0]    # state dimension
    m = Gk.shape[1]    # control dimension
    l = 1   
    d = Ek_F.shape[0]  # disturbance dimension

    # Time horizon
    T = 50
    tol = 1e-6

    # Initialize sequences
    Q = linalg.block_diag(np.eye(n), np.eye(n), tol*np.eye(l))  # example Q matrices
    R = 0.01*np.eye(m)  # example R matrices

    controller = RobustOptimalController(Q, R, alpha)

    # Initial condition
    v0 = np.array([[1], [-1], [0.5], [1], [-1], [0.5], [1]])  # initial state
    v_k = v0
    P_k = linalg.block_diag(np.eye(n), np.eye(n), tol*np.eye(l))  # example Q matrices

    for k in range(100):

        # Step computation
        L_k, K_k, P_k = controller.step(Fk, Gk, Wk, Ek_F, Ek_G, Ek_W, Mk, P_k, Q, R, alpha)

    Sig01 = Ek_F + Ek_G @ K_k[:, :n]
    Sig02 = Ek_W + Ek_G @ K_k[:, n:]

    # print(f"Step {k}:")
    # print(f"L_k: {L_k}")
    print(f"K_k: {K_k}")
    # print(f"Ek_F + Ek_G @ K_k[:, :n]: {Sig01}")
    # print(f"Ek_W + Ek_G @ K_k[:, n:]: {Sig02}")
    # print(f"P_k: {P_k}")

    # Simulate the system
    v_hist, u_hist = simulate_system(v0, alpha, Fk, Gk, Wk, Ek_F, Ek_G, Ek_W, Mk, K_k, T)
    print("Simulation completed.")
    # print("State history:", v_hist)
    # print("Control input history:", u_hist)

    # Plotting the results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(v_hist[:, 0], label='x1')
    plt.plot(v_hist[:, 1], label='x2')
    plt.plot(v_hist[:, 2], label='x3')
    plt.grid()
    plt.title('State History')
    plt.xlabel('Time step')
    plt.ylabel('State values')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(u_hist[:, 0], label='u1')
    plt.plot(u_hist[:, 1], label='u2')
    plt.grid()
    plt.title('Control Input History')
    plt.xlabel('Time step')
    plt.ylabel('Control input values')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # # Define N random initial conditions
    # N = 10
    # n = Fk.shape[0]
    # l = Wk.shape[1]
    # V0 = np.random.randn(N, n + l)  # random initial conditions

    # # Simulate
    # V_hist, U_hist = simulate_multiple_systems(V0, Fk, Gk, Wk, Ek_F, Ek_G, Ek_W, Mk, K_k, T)

    # # Plot first state of each trajectory
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 6))
    # for i in range(N):
    #     plt.plot(V_hist[i, :, 0], label=f'Traj {i+1} (x1)')
    #     plt.plot(V_hist[i, :, 1], label=f'Traj {i+1} (x2)')
    #     plt.plot(V_hist[i, :, 2], label=f'Traj {i+1} (x3)')
    # plt.title('First state evolution across different initial conditions')
    # plt.xlabel('Time step')
    # plt.grid()
    # plt.ylabel('Dynamic state')
    # plt.legend()
    # plt.tight_layout()

    # plt.figure(figsize=(12, 6))
    # for i in range(N):
    #     plt.plot(U_hist[i, :, 0], label=f'Traj {i+1} (u1)')
    #     plt.plot(U_hist[i, :, 1], label=f'Traj {i+1} (u2)')
    # plt.title('Control input evolution across different initial conditions')
    # plt.xlabel('Time step')
    # plt.grid()
    # plt.ylabel('Control Action')
    # plt.legend()
    # plt.tight_layout()

    # plt.show()

if __name__ == "__main__":
    main()
