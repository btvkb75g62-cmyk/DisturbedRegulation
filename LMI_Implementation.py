import numpy as np
import cvxpy as cp

def design_polytopic_h_infinity_controller(vertices, gamma):
    """
    Designs a robust H-infinity state-feedback controller for a discrete-time
    system with polytopic uncertainty.

    The system matrices (F, G) belong to a polytope defined by its vertices:
    (F, G) in Co{(F_1, G_1), ..., (F_L, G_L)}

    The controller u_k = K * x_k is designed to stabilize the system for all
    possible (F, G) in the polytope and guarantee that the H-infinity norm
    from the disturbance w to the performance output z = [x^T, u^T]^T is
    less than gamma.

    Args:
        vertices (list): A list of tuples (F_i, G_i), where F_i and G_i are
                         the numpy arrays for each vertex of the polytope.
        gamma (float): The desired H-infinity performance level.

    Returns:
        np.ndarray: The state-feedback gain matrix K, or None if the
                    problem is infeasible.
    """
    if not vertices:
        raise ValueError("The list of vertices cannot be empty.")

    # Get dimensions from the first vertex
    F1, G1 = vertices[0]
    n, m = G1.shape

    # LMI variables
    # X = P^-1, where P is the common Lyapunov matrix
    X = cp.Variable((n, n), symmetric=True)
    # Y = K * X, where K is the controller gain
    Y = cp.Variable((m, n))

    # We need to solve a set of LMIs, one for each vertex of the polytope.
    constraints = [X >> 0]

    # Performance output is z = [x^T, u^T]^T, which means C1 = [I; 0] and D12 = [0; I]
    # The disturbance input matrix is assumed to be the identity matrix, B1 = I.
    for Fi, Gi in vertices:
        # Check dimensions for consistency
        if Fi.shape != (n, n) or Gi.shape != (n, m):
            raise ValueError("All vertices must have matrices of the same dimensions.")

        # LMI for the i-th vertex based on the Bounded Real Lemma
        # This LMI ensures robust stability and H-infinity performance.
        LMI_i = cp.bmat([
            [-X, np.zeros((n, n)), (Fi @ X + Gi @ Y).T, X, Y.T],
            [np.zeros((n, n)), -gamma**2 * np.eye(n), np.eye(n), np.zeros((n, n)), np.zeros((n, m))],
            [Fi @ X + Gi @ Y, np.eye(n), -X, np.zeros((n, n)), np.zeros((n, m))],
            [X, np.zeros((n, n)), np.zeros((n, n)), -np.eye(n), np.zeros((n, m))],
            [Y, np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, n)), -np.eye(m)]
        ])
        constraints.append(LMI_i << 0)

    # Problem definition
    # We are looking for a feasible solution, so the objective can be set to 0.
    objective = cp.Minimize(0)
    problem = cp.Problem(objective, constraints)

    # Solve the LMI problem. Using the SCS solver is a good default.
    problem.solve(solver=cp.SCS)

    # Check for a valid solution and compute the controller gain K
    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        # If a feasible solution is found, recover the controller gain K = Y * X^-1
        K = Y.value @ np.linalg.inv(X.value)
        return K
    else:
        # If no solution is found, the problem is infeasible for the given gamma.
        return None


def design_attractive_ellipsoid_controller(vertices, W, alpha):
    """
    Designs a robust state-feedback controller for a discrete-time
    system with polytopic uncertainty using the Attractive Ellipsoid Method.

    This implementation is based on Equation (17) from the paper:
    "Robust Disturbance Rejection by the Attractive Ellipsoid Method
     Part II: Discrete-time Systems" by García and Ampountolas (2018).

    Args:
        vertices (list): A list of tuples (A_i, B_i), where A_i and B_i are
                         the numpy arrays for each vertex of the polytope.
        W (np.ndarray): The weighting matrix for the bounded disturbance.
        alpha (float): A design parameter in the interval (0, 1].

    Returns:
        tuple: A tuple containing the state-feedback gain matrix K,
               a list of the Lyapunov matrices P_i, and the minimized
               trace value eta. Returns (None, None, None) if infeasible.
    """
    if not vertices:
        raise ValueError("The list of vertices cannot be empty.")
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in the interval (0, 1].")

    # Get dimensions from the first vertex
    A1, B1 = vertices[0]
    n, m = B1.shape
    num_vertices = len(vertices)

    # --- LMI Variables ---
    # P_i: List of symmetric positive definite Lyapunov matrices, one for each vertex.
    P_list = [cp.Variable((n, n), symmetric=True) for _ in range(num_vertices)]
    # L = K * G, where K is the controller gain.
    L = cp.Variable((m, n))
    # G: A non-singular matrix used as a change of variables.
    G = cp.Variable((n, n))
    # eta: A scalar variable for the optimization objective.
    eta = cp.Variable()

    # --- LMI Constraints ---
    constraints = []
    for i in range(num_vertices):
        Ai, Bi = vertices[i]

        # Check dimensions for consistency
        if Ai.shape != (n, n) or Bi.shape != (n, m):
            raise ValueError("All vertices must have matrices of the same dimensions.")

        # This is the direct implementation of Equation (17) from the paper.
        LMI_i = cp.bmat([
            [P_list[i] - G - G.T,    np.zeros((n, n)),     (G.T @ Ai.T + L.T @ Bi.T),   G.T],
            [np.zeros((n, n)),       -alpha * W,           np.eye(n),                  np.zeros((n, n))],
            [(Ai @ G + Bi @ L),      np.eye(n),           -P_list[i],                  np.zeros((n, n))],
            [G,                      np.zeros((n, n)),     np.zeros((n, n)),            -(1/alpha) * P_list[i]]
        ])
        
        # The LMI must be negative definite for each vertex.
        constraints.append(LMI_i << 0)
        
        # Constraint for the trace minimization (Equation 18).
        constraints.append(cp.trace(P_list[i]) <= eta)
        
        # Ensure P_i is positive definite.
        constraints.append(P_list[i] >> 0)

    # --- Optimization Problem (Equation 19) ---
    # The objective is to minimize the size of the ellipsoid, which is
    # achieved by minimizing the trace of the Lyapunov matrices.
    objective = cp.Minimize(eta)
    problem = cp.Problem(objective, constraints)

    # Solve the Semidefinite Program (SDP).
    # Using SCS solver is a good default for this kind of problem.
    problem.solve(solver=cp.SCS, verbose=False)

    # --- Results ---
    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        # If a feasible solution is found, recover the controller gain K = L * G^-1.
        # We need to ensure G is invertible.
        try:
            G_val = G.value
            G_inv = np.linalg.inv(G_val)
            K = L.value @ G_inv
            P_values = [P.value for P in P_list]
            return K, P_values, eta.value
        except np.linalg.LinAlgError:
            print("Warning: Matrix G is singular. Cannot compute controller gain K.")
            return None, None, None
    else:
        # If no solution is found, the problem is infeasible.
        return None, None, None

if __name__ == '__main__':
    # Example system with polytopic uncertainty defined by 2 vertices.
    # Vertex 1
    F1 = np.array([[ 0.99994561,  0.,         -0.0981,      0.        ],
                    [ 0.,          1.,          0.,          0.05      ],
                    [ 0.,          0.,          1.,          1.        ],
                    [ 0.,          0.,          0.,          1.        ]]) + \
    np.array([[3.69852e-06, 0.00000e+00, 0.00000e+00, 0.00000e+00],
                [0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00],
                [0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00],
                [0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00]])
    
    G1 = np.array([[2.00000000e-05, 0.00000000e+00, 0.00000000e+00],
                   [0.00000000e+00, 2.00000000e-05, 0.00000000e+00],
                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                   [0.00000000e+00, 0.00000000e+00, 1.36612022e-05]]) + \
         np.array([[-1.36e-06,  0.00e+00,  0.00e+00],
                    [ 0.00e+00, -1.36e-06,  0.00e+00],
                    [ 0.00e+00,  0.00e+00,  0.00e+00],
                    [ 0.00e+00,  0.00e+00,  0.00e+00]])

    # Vertex 2 (a perturbation of vertex 1)
    F2 = np.array([[ 0.99994561,  0.,         -0.0981,      0.        ],
                    [ 0.,          1.,          0.,          0.05      ],
                    [ 0.,          0.,          1.,          1.        ],
                    [ 0.,          0.,          0.,          1.        ]]) - \
    np.array([[3.69852e-06, 0.00000e+00, 0.00000e+00, 0.00000e+00],
                [0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00],
                [0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00],
                [0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00]])
    
    G2 = np.array([[2.00000000e-05, 0.00000000e+00, 0.00000000e+00],
                   [0.00000000e+00, 2.00000000e-05, 0.00000000e+00],
                   [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                   [0.00000000e+00, 0.00000000e+00, 1.36612022e-05]]) - \
         np.array([[-1.36e-06,  0.00e+00,  0.00e+00],
                    [ 0.00e+00, -1.36e-06,  0.00e+00],
                    [ 0.00e+00,  0.00e+00,  0.00e+00],
                    [ 0.00e+00,  0.00e+00,  0.00e+00]])
    
    # List of vertices defining the polytope
    polytope_vertices = [(F1, G1), (F2, G2)]

    # Desired H-infinity performance level
    gamma = 33

    # Design the ellipsoid controller
    alpha = 0.1  # Design parameter for the ellipsoid method

    W = np.array([[0/(5**2), 0, 0, 0],
                  [0, 1, 0, 0], 
                  [0, 0, 1/(np.deg2rad(10)**2), 0],
                  [0, 0, 0, 1]]) 

    Kelip, P_list, eta = design_attractive_ellipsoid_controller(polytope_vertices, W, alpha)

    if Kelip is not None:
        print("Successfully designed the robust state-feedback controller.")
        print("State-feedback gain K:")
        print(Kelip)
        print("Minimized trace value eta:", eta)
    else:
        print(f"Failed to design the controller for alpha = {alpha}. The LMI is infeasible.")
        print("Consider adjusting alpha or the polytope vertices to relax the performance requirement.")

    # Design the controller
    Kinf = design_polytopic_h_infinity_controller(polytope_vertices, gamma)

    # if Kinf is not None:
    #     print("Successfully designed the robust H-infinity controller.")
    #     print("State-feedback gain K:")
    #     print(Kinf)
    # else:
    #     print(f"Failed to design the controller for gamma = {gamma}. The LMI is infeasible.")
    #     print("Consider increasing gamma to relax the performance requirement.")

