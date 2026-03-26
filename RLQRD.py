""""
Author: Carlos Persiani
Email: carlospersiani@usp.br / carlospersiani@gmail.com
License: MIT License
"""

import numpy as np
from scipy import linalg

class MRLQRD(object):
    """Markovian Robust Linear Quadratic Regulator with Disturbances | Two Chains Version"""
    def __init__(self, F, G, W, E_F, E_G, E_W, M, Alpha, Q, R, \
                 Prob1=np.array([[1]]), Prob2=np.array([[1]]), mu=1e9, beta=1.02, tol = 1e-4):

        self.F = F
        self.G = G
        self.W = W
        self.E_F = E_F
        self.E_G = E_G
        self.E_W = E_W
        self.M = M
        self.Alpha = Alpha
        self.Q = Q
        self.R = R

        if not isinstance(F, list):
            self.F = [F for _ in range(Prob1.shape[0])]
            self.F = np.array(self.F)
        if not isinstance(G, list):
            self.G = [G for _ in range(Prob1.shape[0])]
            self.G =  np.array(self.G)
        if not isinstance(W, list):
            self.W = [W for _ in range(Prob1.shape[0])]
            self.W = np.array(self.W)
        if not isinstance(E_F, list):
            self.E_F = [E_F for _ in range(Prob1.shape[0])]
            self.E_F = np.array(self.E_F)
        if not isinstance(E_G, list):
            self.E_G = [E_G for _ in range(Prob1.shape[0])]
            self.E_G = np.array(self.E_G)
        if not isinstance(E_W, list):
            self.E_W = [E_W for _ in range(Prob1.shape[0])]
            self.E_W = np.array(self.E_W)
        if not isinstance(Q, list):
            self.Q = [Q for _ in range(Prob1.shape[0])]
            self.Q = np.array(self.Q)
        if not isinstance(R, list):
            self.R = [R for _ in range(Prob1.shape[0])]
            self.R = np.array(self.R)
        if not isinstance(M, list):
            M = [M for _ in range(Prob1.shape[0])]
            M = np.array(M)

        self.Prob1 = Prob1
        self.Prob2 = Prob2

        self.modes1 = Prob1.shape[0]
        self.modes2 = Prob2.shape[0]

        self.M_stili = [np.vstack([M[i], np.zeros((self.W[i].shape[1], M[i].shape[1]))]) for i in range(Prob1.shape[0])]
        self.H = [np.zeros_like((len(M[i]), self.F[i].shape[0]+self.W[i].shape[1]+M[i].shape[1]+self.W[i].shape[1])) for i in range(Prob1.shape[0])]

        self.mu = mu
        self.beta = beta
        self.lambd = [self.beta * np.linalg.norm(self.mu * self.M_stili[i].T @ self.M_stili[i], ord=2)**2 for i in range(len(self.M_stili))]

        self.Q = [[linalg.block_diag(self.Q[i], tol * np.eye(self.W[i].shape[1])) for i in range(len(self.Prob1))] for _ in range(len(self.Prob2))]
        self.R = [[self.R[i] for i in range(len(self.Prob1))] for _ in range(len(self.Prob2))]
        self.P = [[linalg.block_diag(np.eye(F[0].shape[0]), tol * np.eye(self.W[0].shape[1])) for _ in range(len(self.Prob1))] for _ in range(len(self.Prob2))]

        if np.isscalar(Alpha):
            self.Alpha = Alpha * np.eye(self.W.shape[1])
        else:
            self.Alpha = Alpha

        if not isinstance(Alpha, list):
            self.Alpha = [self.Alpha for _ in range(Prob2.shape[0])]
            self.Alpha = np.array(self.Alpha)

        self.R2 = []

    def compute_matrices(self):

        n = self.F[0].shape[1]
        l = self.Alpha[0].shape[1]

        G_aug = [np.vstack([self.G[i], np.zeros((l, self.G[i].shape[1]))]) for i in range(self.modes1)]

        self.E_F_aug = [np.hstack([self.E_F[i], self.E_W[i]]) for i in range(len(self.Prob1))]
        self.E_G_aug = self.E_G

        bar_I = np.vstack([
            np.eye(n + l),
            np.zeros((self.E_G_aug[0].shape[0], n + l))
        ])

        bar_G = [np.vstack([G_aug[i], self.E_G_aug[i]]) for i in range(self.modes1)]

        Sigma_k = [np.block([[self.mu**(-1) * np.eye(self.M_stili[i].shape[0]) - self.lambd[i]**(-1) * self.M_stili[i] @ self.M_stili[i].T, np.zeros((self.M_stili[i].shape[0], self.E_G[i].shape[0]))],
                            [np.zeros((self.E_G[i].shape[0], self.M_stili[i].shape[0])), self.lambd[i]**(-1) * np.eye(self.E_G[i].shape[0])]]) for i in range(self.modes1)]


        F_aug = [[np.block([
            [self.F[i], self.W[i]],
            [np.zeros((l, n)),  self.Alpha[j]]
        ]) for i in range(self.modes1)] for j in range(self.modes2)]

        bar_F = [[np.vstack([F_aug[j][i], self.E_F_aug[i]]) for i in range(self.modes1)] for j in range(self.modes2)]

        return F_aug, G_aug, bar_I, bar_F, bar_G, Sigma_k
    
    def make_open_Kernel(self, P_inv, R_inv, Q_inv, Sigma_k, bar_I_k, bar_G_k):

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

    def make_open_Right(self, bar_F):
        
        n = self.F[0][0].shape[0]
        m = self.G[0].shape[1]
        l = self.W[0].shape[1]

        row1 =  np.zeros((n + l, n+l))
        row2 =  np.zeros((m, n + l))
        row3 = -np.eye(n+l)
        row4 =  bar_F
        row5 =  np.zeros((n + l, n + l))
        row6 =  np.zeros((m, n + l))

        return np.vstack([row1, row2, row3, row4, row5, row6])

    def make_open_Left(self, bar_F):

        n = self.F[0].shape[0]
        m = self.G[0].shape[1]
        l = self.W[0].shape[1]

        row1 = np.hstack([np.zeros((n + l, n+l + m + n+l + bar_F.T.shape[1] )), np.eye(n+l), np.zeros((n+l, m))])
        row2 = np.hstack([np.zeros((m, n+l + m + n+l + bar_F.T.shape[1] + n+l)), np.eye(m)])
        row3 = np.hstack([np.zeros((n+l, n+l + m)), -np.eye(n+l), bar_F.T, np.zeros((n+l, n+l + m))])

        return np.vstack([row1, row2, row3])

    def control_compensation(self, bar_G, bar_W):

        return - np.linalg.pinv(bar_G.T @ bar_G) @ bar_G.T @ bar_W

    def closed_recursion(self):

        F_aug, G_aug, bar_I, bar_F, bar_G, Sigma_k = self.compute_matrices()

        # self.mode matrices self.P
        P_k = self.P.copy()
        Kk = np.array([[np.zeros((G_aug[i].shape[1], F_aug[j][i].shape[1])) for i in range(self.modes1)] for j in range(self.modes2)])
        Lk = np.array([[np.zeros((F_aug[j][i].shape[1], F_aug[j][i].shape[1])) for i in range(self.modes1)] for j in range(self.modes2)])

        for _ in range(400):

            PSI = np.array([[np.zeros((F_aug[0][0].shape[0], F_aug[0][0].shape[0])) for _ in range(self.modes1)] for _ in range(self.modes2)])

            for k in range(self.modes2):
                for h in range(self.modes2):
                    for i in range(self.modes1):
                        for j in range(self.modes1):
                            PSI[k][i] += float(self.Prob1[i, j]) * float(self.Prob2[k, h]) * P_k[h][j]


            for k in range(self.modes2):
                for i in range(self.modes1):
                    inv_P_k = np.linalg.inv(PSI[k][i])

                    temp = bar_I @ inv_P_k @ bar_I.T + bar_G[i] @ np.linalg.inv(self.R[k][i]) @ bar_G[i].T
                    temp_inv = np.linalg.inv(temp)

                    P_k[k][i] = self.Q[k][i] + bar_F[k][i].T @ temp_inv @ bar_F[k][i]

                    Lk[k][i] = inv_P_k @ bar_I.T @ temp_inv @ bar_F[k][i]
                    Kk[k][i] = - np.linalg.inv(self.R[k][i]) @ bar_G[i].T @ temp_inv @ bar_F[k][i]

            R2 = 0
            for k in range(self.modes2):
                for i in range(self.modes1):
                    R2 += np.linalg.norm(self.E_F_aug[i]  + self.E_G_aug[i] @ Kk[k][i], ord=2)
            self.R2.append(1 / (1 + R2))

        return Lk, Kk, P_k

    def open_recursion(self):

        n = self.F[0].shape[0]
        m = self.G[0].shape[1]
        l = self.W[0].shape[1]

        F_aug, G_aug, bar_I, bar_F, bar_G, Sigma_k = self.compute_matrices()
        P_k = self.P.copy()

        Kk = np.array([[np.zeros((G_aug[i].shape[1], F_aug[k][i].shape[1])) for i in range(self.modes1)] for k in range(self.modes2)])
        Lk = np.array([[np.zeros((F_aug[k][i].shape[1], F_aug[k][i].shape[1])) for i in range(self.modes1)] for k in range(self.modes2)])

        R_inv = [[np.linalg.inv(self.R[k][i]) for i in range(self.modes1)] for k in range(self.modes2)]
        Q_inv = [[np.linalg.inv(self.Q[k][i]) for i in range(self.modes1)] for k in range(self.modes2)]

        for _ in range(400):

            PSI = np.array([[np.zeros((F_aug[k][i].shape[0], F_aug[k][i].shape[0])) for i in range(self.modes1)] for k in range(self.modes2)])
            for k in range(self.modes2):
                for h in range(self.modes2):
                    for i in range(self.modes1):
                        for j in range(self.modes1):
                            PSI[k][i] += float(self.Prob1[i, j]) * float(self.Prob2[k, h]) * P_k[h][j]

            for k in range(self.modes2):
                for i in range(self.modes1):
                    Left = self.make_open_Left(bar_F[k][i])

                    Kernel = self.make_open_Kernel(
                        np.linalg.inv(P_k[k][i]), 
                        R_inv[k][i], 
                        Q_inv[k][i], 
                        Sigma_k[i], 
                        bar_I, 
                        bar_G[i]
                    )

                    Right = self.make_open_Right(bar_F[k][i])

                    solution = Left @ np.linalg.solve(Kernel, Right)

                    P_k[k][i] = solution[n+l + m:n+l + m + n+l, :]

                    Lk[k][i] = solution[:n+l, :]
                    Kk[k][i] = solution[n+l:n+l + m, :]

            R2 = 0
            for k in range(self.modes2):
                for i in range(self.modes1):
                    R2 += np.linalg.norm(self.E_F_aug[i]  + self.E_G_aug[i] @ Kk[k][i], ord=2)
            self.R2.append(1 / (1 + R2))

        return Lk, Kk, P_k

    def variant_compensation(self, solution_type='auto'):

        n = self.F[0].shape[0]
        m = self.G[0].shape[1]
        l = self.W[0].shape[1]

        F_aug, G_aug, bar_I, bar_F, bar_G, Sigma_k = self.compute_matrices()
        P_k = self.P.copy()

        Kkvar = np.array([[np.zeros((G_aug[i].shape[1], F_aug[k][i].shape[1])) for i in range(self.modes1)] for k in range(self.modes2)])

        # Check condition for the existence of such variant controller
        for i in range(self.modes1):
            bar_G_i = np.vstack([self.G[i], self.E_G[i]])
            bar_W_i = np.vstack([self.W[i], self.E_W[i]])
            if np.linalg.matrix_rank(np.hstack([bar_G_i, bar_W_i])) != np.linalg.matrix_rank(bar_G_i):
                print(f"Rank condition not satisfied for mode {i}. Cannot compute variant compensation.")
            if np.linalg.matrix_rank(bar_G_i) < bar_G_i.shape[1]:
                raise ValueError(f"bar_G for mode {i} does not have full column rank. Cannot compute variant compensation.")
            for j in range(self.modes2):    
                Kkvar[j, i,:,-l:] = self.control_compensation(bar_G_i, bar_W_i)

        Lk, Kk, Pk = None, None, None

        self.W = list(np.zeros_like(self.W))
        self.E_W = list(np.zeros_like(self.E_W))
        self.Lambda = list(np.array([[0]]))
        self.Prob2 = np.array([[1]])

        if solution_type == 'auto':

            try:
                Lk, Kk, Pk = self.closed_recursion()
                label = "Closed Recursion Found"
            except np.linalg.LinAlgError:
                print("Closed recursion failed, falling back to open recursion.")
                Lk, Kk, Pk = self.open_recursion()
                label = "Open Recursion Used as Fallback"

        if solution_type == 'closed':
            try:
                Lk, Kk, Pk = self.closed_recursion()
                label = "Closed Recursion Used"
            except np.linalg.LinAlgError:
                print("Closed recursion failed")

        if solution_type == 'open':
            try:
                Lk, Kk, Pk = self.open_recursion()
                label = "Open Recursion Used"
            except np.linalg.LinAlgError:
                print("Open recursion failed")

        try:
            print(label)
        except:
            pass

        for i in range(self.modes1):
            Kk[0, i] += Kkvar[0, i]

        return Lk, Kk, Pk




    def main(self, solution_type='auto'):

        Lk, Kk, Pk = None, None, None

        if solution_type == 'auto':

            try:
                Lk, Kk, Pk = self.closed_recursion()
                label = "Closed Recursion Found"
            except np.linalg.LinAlgError:
                print("Closed recursion failed, falling back to open recursion.")
                Lk, Kk, Pk = self.open_recursion()
                label = "Open Recursion Used as Fallback"

        if solution_type == 'closed':
            try:
                Lk, Kk, Pk = self.closed_recursion()
                label = "Closed Recursion Used"
            except np.linalg.LinAlgError:
                print("Closed recursion failed")

        if solution_type == 'open':
            try:
                Lk, Kk, Pk = self.open_recursion()
                label = "Open Recursion Used"
            except np.linalg.LinAlgError:
                print("Open recursion failed")

        try:
            print(label)
        except:
            pass

        return Lk, Kk, Pk


def main():

    F = [np.array([[1.1, 0, 0], 
                   [0, 0, 1.2],
                   [-1, 1, 0]]),np.array([[1.1, 0, 0], 
                                          [0, 0, 1.2],
                                          [-1, 1, 0]])]  
    
    G = [np.array([[0, 1],
                   [1, 1],
                   [-1, 0]]), np.array([[0, 1],
                                        [1, 1],
                                        [-1, 0]])]  
    
    W = [np.array([[0.2],
                   [-0.5],
                   [0.7]]), 0*np.array([[0.2],
                                      [-0.5],
                                      [0.7]])]  
    
    M = [np.array([[0.7],
                   [0.5],
                   [-0.7]]), np.array([[0.7],
                                       [0.5],
                                       [-0.7]])]

    E_F = [np.array([[0.4, 0.5, -0.6]]), np.array([[0.4, 0.5, -0.6]])]
    E_G = [np.array([[0.4, -0.4]]), np.array([[0.4, -0.4]])]
    E_W = [np.array([[-0.36]]), np.array([[-0.36]])]

    Lambda = [np.array([[0.95]]), 
             np.array([[1.10]])]

    Prob1=np.array([[0.5, 0.5], 
                    [0.5, 0.5]])
    Prob2=np.array([[0.8, 0.2], 
                    [0.8, 0.2]])

    markov = MRLQRD(F, G, W, E_F, E_G, E_W, M, Lambda, Q=np.eye(3), R=np.eye(2), Prob1=Prob1, Prob2=Prob2)
    L, K, P = markov.main(solution_type='auto')
    
    print("K mode (1,1):")
    print(K[0][0])
    print("\nK mode (1,2):")
    print(K[0][1])
    print("\nK mode (2,1):")
    print(K[1][0])
    print("\nK mode (2,2):")
    print(K[1][1])


if __name__ == "__main__":
    main()
