import numpy as np
import scipy.linalg as linalg
from RLQRD import RLQRD

class RobustOptimalController:

    def __init__(self, F, G, W, E_F, E_G, E_W, M, Q_seq, R_seq, alpha_seq, mu=1e9, beta=1.05):

        self.F = F
        self.G = G
        self.W = W
        self.E_F = E_F
        self.E_G = E_G
        self.E_W = E_W
        self.M = M

        self.Q_seq = Q_seq
        self.R_seq = R_seq

        if np.isscalar(alpha_seq): 
            l = W.shape[1]
            self.alpha_seq = alpha_seq * np.eye(l)
        else:
            self.alpha_seq = alpha_seq

        self.mu = mu
        self.beta = beta

    def compute_matrices(self, Fk, Gk, Wk, Ek_F, Ek_G, Ek_W, Mk):

        n = Fk.shape[0]
        l = Wk.shape[1]

        F_aug = np.block([
            [Fk, Wk],
            [np.zeros((l, n)),  self.alpha_seq]
        ])

        G_aug = np.vstack([Gk, np.zeros((l, Gk.shape[1]))])

        E_F_aug = np.hstack([Ek_F, Ek_W])
        E_G_aug = Ek_G
        
        bar_I = np.vstack([
            np.eye(n + l),
            np.zeros((E_G_aug.shape[0], n + l))
        ])
        bar_F = np.vstack([F_aug, E_F_aug])
        bar_G = np.vstack([G_aug, E_G_aug])

        return F_aug, G_aug, bar_I, bar_F, bar_G

    def step(self, Fk, Gk, Wk, Ek_F, Ek_G, Ek_W, Mk, P_k, Qk, Rk):

        bar_I, bar_F, bar_G = self.compute_matrices(Fk, Gk, Wk, Ek_F, Ek_G, Ek_W, Mk)

        inv_P_k = np.linalg.inv(P_k)

        temp = bar_I @ inv_P_k @ bar_I.T + bar_G @ np.linalg.inv(Rk) @ bar_G.T
        temp_inv = np.linalg.inv(temp)

        Lk = inv_P_k @ bar_I.T @ temp_inv @ bar_F 
        Kk = - np.linalg.inv(Rk) @ bar_G.T @ temp_inv @ bar_F 

        P_k = Qk + bar_F.T @ temp_inv @ bar_F

        return Lk, Kk, P_k

    def make_block_matrix_Bk(self, P_inv, R_inv, Q_inv, Sigma_k, bar_I_k, bar_G_k):

        n = P_inv.shape[0]
        m = R_inv.shape[0]
        q = Q_inv.shape[0]
        s = Sigma_k.shape[0]

        # Check dimensions
        assert bar_I_k.shape[0] == s, "bar_I_k row count must match Sigma_k"
        assert bar_G_k.shape[0] == s, "bar_G_k row count must match Sigma_k"

        # Build rows ensuring consistent column count
        # The structure should be: [P_inv | 0 | 0 | 0 | I | 0]
        #                          [0 | R_inv | 0 | 0 | 0 | I]
        #                          [0 | 0 | Q_inv | 0 | 0 | 0]
        #                          [0 | 0 | 0 | Sigma_k | bar_I_k | -bar_G_k]
        #                          [I | 0 | 0 | bar_I_k.T | 0 | 0]
        #                          [0 | I | 0 | -bar_G_k.T | 0 | 0]
        
        row1 = np.hstack([P_inv, np.zeros((n, m)), np.zeros((n, q)), np.zeros((n, s)), np.eye(n), np.zeros((n, m))])
        row2 = np.hstack([np.zeros((m, n)), R_inv, np.zeros((m, q)), np.zeros((m, s)), np.zeros((m, n)), np.eye(m)])
        row3 = np.hstack([np.zeros((q, n)), np.zeros((q, m)), Q_inv, np.zeros((q, s)), np.zeros((q, n)), np.zeros((q, m))])
        row4 = np.hstack([np.zeros((s, n)), np.zeros((s, m)), np.zeros((s, q)), Sigma_k, bar_I_k, -bar_G_k])
        row5 = np.hstack([np.eye(n), np.zeros((n, m)), np.zeros((n, q)), bar_I_k.T, np.zeros((n, n)), np.zeros((n, m))])
        row6 = np.hstack([np.zeros((m, n)), np.eye(m), np.zeros((m, q)), -bar_G_k.T, np.zeros((m, n)), np.zeros((m, m))])

        return np.vstack([row1, row2, row3, row4, row5, row6])

    def compute_recursive_step(self, P_k_plus_1, F_k, G_k, Q_k, R_k, E_F_k, E_G_k, H_k, mu, beta_k):
        """
        Computes one step of the recursive solution for a robust LQR-like problem.

        This function implements the equation:
        [L_k; K_k; P_k] = M1 @ inv(M2) @ M3
        by solving the more stable linear system M2 @ x = M3 and then extracting the results.

        Args:
            P_k_plus_1 (np.ndarray): The matrix P from step k+1.
            F_k (np.ndarray): The state transition matrix.
            G_k (np.ndarray): The input matrix.
            Q_k (np.ndarray): The state cost matrix.
            R_k (np.ndarray): The input cost matrix.
            E_F_k (np.ndarray): Uncertainty matrix related to F_k.
            E_G_k (np.ndarray): Uncertainty matrix related to G_k.
            H_k (np.ndarray): Uncertainty weighting matrix.
            mu (float): A positive scalar parameter.
            lambda_k (float): A positive scalar regularization parameter.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the
            calculated matrices (L_k, K_k, P_k).
        """
        # --- 1. Determine dimensions from inputs ---
        dim_n = F_k.shape[1]
        dim_m = G_k.shape[1]
        dim_l = self.W.shape[1] 
        dim_s = E_F_k.shape[0] + F_k.shape[0] # Dimension of the augmented dynamics

        # --- 2. Form composite and intermediate matrices ---
        bar_F_k = np.vstack([F_k, E_F_k])
        bar_G_k = np.vstack([G_k, E_G_k])

        # Form Sigma_k
        HHT = H_k @ H_k.T
        dim_h = HHT.shape[0]

        lambda_k = beta_k * np.linalg.norm(mu * HHT, ord=2)  # Regularization parameter

        sigma_top_left = (1/mu) * np.eye(dim_h) - (1/lambda_k) * HHT
        sigma_bottom_right = (1/lambda_k) * np.eye(dim_s - dim_h)
        
        Sigma_k = np.block([
            [sigma_top_left,               np.zeros((dim_h, dim_s - dim_h))],
            [np.zeros((dim_s - dim_h, dim_h)), sigma_bottom_right]
        ])

        # Form bar_I_k based on the structure in the provided equation.
        # The I_{n+1} in the paper likely refers to the state dimension, which we take as dim_n.
        bar_I_k = np.vstack([np.eye(dim_n+dim_l), np.zeros((dim_s - dim_n, dim_n))])

        # --- 3. Construct the large matrix M2 to be inverted ---
        P_inv = np.linalg.inv(P_k_plus_1)
        R_inv = np.linalg.inv(R_k)
        Q_inv = np.linalg.inv(Q_k)

        M2 = np.block([
            [P_inv,              np.zeros((dim_n, dim_m)), np.zeros((dim_n, dim_n)), np.zeros((dim_n, dim_s)), np.eye(dim_n),       np.zeros((dim_n, dim_m))],
            [np.zeros((dim_m, dim_n)), R_inv,              np.zeros((dim_m, dim_n)), np.zeros((dim_m, dim_s)), np.zeros((dim_m, dim_n)), np.eye(dim_m)      ],
            [np.zeros((dim_n, dim_n)), np.zeros((dim_n, dim_m)), Q_inv,              np.zeros((dim_n, dim_s)), np.zeros((dim_n, dim_n)), np.zeros((dim_n, dim_m))],
            [np.zeros((dim_s, dim_n)), np.zeros((dim_s, dim_m)), np.zeros((dim_s, dim_n)), Sigma_k,           bar_I_k,              -bar_G_k          ],
            [np.eye(dim_n),      np.zeros((dim_n, dim_m)), np.zeros((dim_n, dim_n)), bar_I_k.T,         np.zeros((dim_n, dim_n)), np.zeros((dim_n, dim_m))],
            [np.zeros((dim_m, dim_n)), np.eye(dim_m),      np.zeros((dim_m, dim_n)), -bar_G_k.T,        np.zeros((dim_m, dim_n)), np.zeros((dim_m, dim_m))]
        ])

        # --- 4. Construct the vector M3 ---
        M3 = np.vstack([
            np.zeros((dim_n, dim_n)),
            np.zeros((dim_m, dim_n)),
            -np.eye(dim_n),
            bar_F_k,
            np.zeros((dim_n, dim_n)),
            np.zeros((dim_m, dim_n))
        ])
        
        # --- 5. Solve the linear system M2 @ x = M3 for the solution vector x ---
        # This is more numerically stable than computing the inverse of M2 directly.
        # Note: M3 has multiple columns, so the solution 'x' will also have multiple columns.
        solution_vec = np.linalg.solve(M2, M3)

        # --- 6. Partition the solution vector 'x' to extract results ---
        # The partitions correspond to the block-row structure of M2.
        idx = [dim_n, dim_m, dim_n, dim_s, dim_n, dim_m]
        cum_idx = np.cumsum(idx)
        
        # x1 = solution_vec[0:cum_idx[0], :]
        # x2 = solution_vec[cum_idx[0]:cum_idx[1], :]
        x3 = solution_vec[cum_idx[1]:cum_idx[2], :]
        x4 = solution_vec[cum_idx[2]:cum_idx[3], :]
        x5 = solution_vec[cum_idx[3]:cum_idx[4], :]
        x6 = solution_vec[cum_idx[4]:cum_idx[5], :]
        
        # --- 7. Calculate L_k, K_k, and P_k from the partitions ---
        # This logic comes from multiplying the solution by the M1 matrix.
        L_k = x5
        K_k = x6
        P_k = -x3 + bar_F_k.T @ x4
        
        return L_k, K_k, P_k

    def compute_recursive_open_solution(self, P_k, R, Q, barI, barF, barG, H, mu=1e8, beta=1.05):

        m = self.G.shape[1]
        n = self.F.shape[1]

        R_inv_k = np.linalg.inv(R)
        Q_inv_k = np.linalg.inv(Q)
        P_next_inv = np.linalg.inv(P_k)

        lambd = beta * np.linalg.norm(mu * H @ H.T, ord=2)  # Regularization parameter

        Sigma_k = np.block([[self.mu**(-1) * np.eye(H.shape[0]) - lambd**(-1) * H @ H.T, np.zeros((H.shape[0], self.E_G.shape[0]))],
                             [np.zeros((self.E_G.shape[0], H.shape[0])), lambd**(-1) * np.eye(self.E_G.shape[0])]])

        bar_I_k, bar_F_k, bar_G_k = self.compute_matrices(self.F, self.G, self.W, self.E_F, self.E_G, self.E_W, self.M)

        Left_k = np.block([np.zeros((n, n)), np.zeros((n, m)), np.zeros((n, n)), np.zeros((n, Sigma_k.shape[0])), np.eye(n), np.zeros((n, m))])

        B_k = self.make_block_matrix_Bk(P_next_inv, R_inv_k, Q_inv_k, Sigma_k, bar_I_k, bar_G_k)

        Right_k = np.vstack([
                    np.zeros((n, n)),
                    np.zeros((m, n)),
                    -np.eye(n),
                    bar_F_k,
                    np.zeros((n, n)),
                    np.zeros((m, n))
                ])
        
        P_k_new = np.linalg.solve(B_k, Right_k)

        return L_k, K_k, P_k_new

    def main(self, mu=1e10, beta=1.05, solution_type='auto'):

        self.mu = mu
        self.beta = beta
        
        P_k = linalg.block_diag(np.eye(self.F.shape[0]), 1e-6 * np.eye(self.W.shape[1]))

        if solution_type == 'auto':
            # First try the recursive Closed Solution, when mu -> infty
            try:

                for _ in range(400):
                    L_k, K_k, P_k = self.step(self.F, self.G, self.W, self.E_F, self.E_G, self.E_W, self.M, P_k, self.Q_seq, self.R_seq)

                print("Closed Solution computed successfully.")

            # If the closed solution fails due to singular matrix, we use the recursive open solution
            except np.linalg.LinAlgError as e:

                barI, barF, barG = self.compute_matrices(self.F, self.G, self.W, self.E_F, self.E_G, self.E_W, self.M)

                for _ in range(400):
                    L_k, K_k, P_k = self.compute_recursive_open_solution(P_k, self.R_seq, self.Q_seq, barI, barF, barG, self.M, mu=self.mu, beta=self.beta)
                
                print("Open Solution used due to singular matrix in Closed Solution.")

        if solution_type == 'open':

            F_aug, G_aug, bar_I, bar_F, bar_G = self.compute_matrices(self.F, self.G, self.W, self.E_F, self.E_G, self.E_W, self.M)

            for _ in range(400):

                #L_k, K_k, P_k = self.compute_recursive_open_solution(P_k, self.R_seq, self.Q_seq, barI, barF, barG, self.M, mu=self.mu, beta=self.beta)

                L_k, K_k, P_k = self.compute_recursive_step(P_k, F_aug, G_aug, self.Q_seq, self.R_seq, self.E_F, self.E_G, self.M, mu=self.mu, beta_k=self.beta)

            print("Open Solution computed successfully.")

        elif solution_type == 'closed':

            try:

                for _ in range(400):
                    L_k, K_k, P_k = self.step(self.F, self.G, self.W, self.E_F, self.E_G, self.E_W, self.M, P_k, self.Q_seq, self.R_seq)

                print("Closed Solution computed successfully.")

            except np.linalg.LinAlgError as e:
                
                print("Closed solution failed due to singular matrix.")

        return L_k, K_k, P_k

def simulate_system(v0, alpha, Fk, Gk, Wk, Ek_F, Ek_G, Ek_W, Mk, K_k, K_lqr, KRLQR, T, tol=1e-6):
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
    xlqr = v0[:n]  # initial state
    xrlqr = v0[:n]  # initial state for LQR
    phi_k = v0[-1:]  # initial disturbance

    v_hist = [v_k.flatten()]
    x_lqr_hist = [v_k[:n].flatten()]
    x_rlqr_hist = [v_k[:n].flatten()]
    u_hist = []
    disturbance_hist = []

    for k in range(T):

        # Step 2: Compute control input
        u_k = K_k @ v_k
        u_hist.append(u_k.flatten())

        u_lqr = K_lqr @ xlqr

        u_rlqr = KRLQR @ xrlqr

        # Step 3: Sample an uncertainty Delta_k (bounded)
        d = Mk.shape[1] if Mk.ndim > 1 else 1  # disturbance dimension
        #Delta_k = np.random.uniform(-1, 1, (d, d))  # simple uncertainty (could be more sophisticated)
        # make a Delta_k matrix a sine wave using the simulation time k with the shape of (d, d)
        #Delta_k = np.sin(k / (0.8*2*np.pi)) * np.ones((d, d))
        Delta_k = np.random.uniform(-1, 1, (d, d))

        # Step 4: Compute uncertain matrices
        delta_Fk = Mk @ Delta_k @ Ek_F
        delta_Gk = Mk @ Delta_k @ Ek_G
        delta_Wk = Mk @ Delta_k @ Ek_W

        # Step 5: Update system state
        x_k = v_k[:n]

        disturbance = (Wk + delta_Wk) * phi_k

        x_k1 = (Fk + delta_Fk) @ x_k + (Gk + delta_Gk) @ u_k + disturbance
        phi_k = alpha*phi_k

        xlqr = (Fk + delta_Fk) @ xlqr + (Gk + delta_Gk) @ u_lqr + disturbance

        xrlqr = (Fk + delta_Fk) @ xrlqr + (Gk + delta_Gk) @ u_rlqr + disturbance

        v_k = np.vstack((x_k1, phi_k))
        v_hist.append(v_k.flatten())
        x_lqr_hist.append(xlqr.flatten())
        x_rlqr_hist.append(xrlqr.flatten())
        disturbance_hist.append(disturbance.flatten())

    v_hist = np.array(v_hist)
    u_hist = np.array(u_hist)
    x_lqr_hist = np.array(x_lqr_hist)
    x_rlqr_hist = np.array(x_rlqr_hist)
    disturbance_hist = np.array(disturbance_hist)

    return v_hist, u_hist, disturbance_hist, x_lqr_hist, x_rlqr_hist

def main():

    # Example matrices for timestep k
    Fk = np.array([[1.1, 0, 0], 
                   [0, 0, 1.2],
                   [-1, 1, 0]])  # state transition matrix
    Gk = np.array([[0, 1],
                   [1, 1],
                   [-1, 0]])  # control input matrix
    Wk = np.array([[0.2],
                   [-0.5],
                   [0]])  # disturbance vector
    alpha = 0.99  # decay factor
    
    Ek_F = np.array([[0.4, 0.5, -0.6]])
    Ek_G = np.array([[0.4, -0.4]])
    Ek_W = np.array([[0.5]])

    Mk = np.array([[0.7],
                   [0.5],
                   [-0.7]])  # disturbance input matrix

    # Dimensions
    n = Fk.shape[0]    # state dimension
    m = Gk.shape[1]    # control dimension
    l = 1   
    d = Ek_F.shape[0]  # disturbance dimension

    # Time horizon
    T = 400
    tol = 1e-6

    # Initialize sequences
    Q = linalg.block_diag(np.eye(n))  # example Q matrices
    R = np.eye(m)  # example R matrices

    ## Standard LQR controller
    P_stardard = linalg.solve_discrete_are(Fk, Gk, Q[:n, :n], R)
    K_lqr = - np.linalg.inv(R + Gk.T @ P_stardard @ Gk) @ (Gk.T @ P_stardard @ Fk)

    ## H Infinity controller
    K_Hinf = np.array([[ 0.19955593,  0.4307009,  -0.68389327],
                       [-1.82730629,  0.34085714,  0.41112107]])

    controller = RLQRD(Fk, Gk, Wk, Ek_F, Ek_G, Ek_W, Mk, alpha, Q, R)
    controllerRLQR = RLQRD(Fk, Gk, 0*Wk, Ek_F, Ek_G, 0*Ek_W, Mk, alpha, Q, R)

    # Initial condition
    v0 = np.array([[1], [-1], [0.5], [1]])  # initial state
    v_k = v0
    P_k = linalg.block_diag(np.eye(n), tol*np.eye(l))  # example Q matrices

    L_k, K_k, P_k = controller.main(solution_type='auto')
    LRLQR, KRLQR, PRLQR = controllerRLQR.main()

    Sig01 = Ek_F + Ek_G @ K_k[:, :n]
    Sig02 = Ek_W + Ek_G @ K_k[:, n:]

    print(f"L_k: {L_k}")
    print(f"K_k: {K_k}")
    print(f"Ek_F + Ek_G @ K_k[:, :n]: {Sig01}")
    print(f"Ek_W + Ek_G @ K_k[:, n:]: {Sig02}")
    print(f"P_k: {P_k}")


    # Simulate the system
    v_hist, u_hist, dist, x_lqr, x_rlqr = simulate_system(v0, alpha, Fk, Gk, Wk, Ek_F, Ek_G, Ek_W, Mk, K_k, K_lqr, KRLQR[:, :n], T)
    print("Simulation completed.")

    # Plotting the results
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.subplot(3, 1, 1)

    # plt.plot(v_hist[:, 0], 'b-', label=r'$x_{1}$ RLQRBD')
    # plt.plot(v_hist[:, 1], 'k-', label=r'$x_{1}$ RLQRBD')
    # plt.plot(v_hist[:, 2], 'r-', label=r'$x_{1}$ RLQRBD')
    plt.plot(np.linalg.norm(v_hist[:, :n], axis=1), 'b-', label=r'$||x||^{2}$ RLQRD')

    # plt.plot(x_lqr[:, 0], 'b--', label=r'$x_{1}$ LQR')
    # plt.plot(x_lqr[:, 1], 'k--', label=r'$x_{2}$ LQR')
    # plt.plot(x_lqr[:, 2], 'r--', label=r'$x_{3}$ LQR')
    plt.plot(np.linalg.norm(x_lqr, axis=1), 'r--', label=r'$||x||^{2}$ LQR')

    # plt.plot(x_rlqr[:, 0], 'b:', label=r'$x_{1}$ RLQR')
    # plt.plot(x_rlqr[:, 1], 'k:', label=r'$x_{2}$ RLQR')
    # plt.plot(x_rlqr[:, 2], 'r:', label=r'$x_{3}$ RLQR')
    plt.plot(np.linalg.norm(x_rlqr, axis=1), 'k:', label=r'$||x||^{2}$ RLQR')

    plt.grid()
    plt.title('State Space')
    #plt.xlabel('Time step')
    plt.ylabel('State values [ ]')
    plt.ylim(bottom=0)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(u_hist[:, 0], 'r', label=r'$u_1$')
    plt.plot(u_hist[:, 1], 'k', label=r'$u_2$')
    plt.grid()
    plt.title('Control Action')
    #plt.xlabel('Time step')
    plt.ylabel('Control Action [ ]')
    plt.legend()
    plt.tight_layout()

    plt.subplot(3, 1, 3)
    plt.plot(dist[:, 0], 'b', label=r'$(w_{1}+\delta w_{1})\varphi$')
    plt.plot(dist[:, 1], 'k', label=r'$(w_{2}+\delta w_{2})\varphi$')
    plt.plot(dist[:, 2], 'r', label=r'$(w_{3}+\delta w_{3})\varphi$')
    plt.grid()
    plt.title(r'Disturbance | $\alpha_k = 0.99$')
    plt.xlabel('Time step')
    plt.ylabel('Disturbance [ ]')
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
