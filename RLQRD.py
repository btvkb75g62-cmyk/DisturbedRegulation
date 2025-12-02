import numpy as np
from scipy import linalg


class RLQRD(object):
    """Robust Linear Quadratic Regulator with Disturbances"""
    def __init__(self, F, G, W, E_F, E_G, E_W, M, Alpha, Q, R, mu=1e9, beta=1.02, tol = 1e-4):

        self.F = F
        self.G = G
        self.W = W
        self.E_F = E_F
        self.E_G = E_G
        self.E_W = E_W
        self.M = M

        self.M_stili = np.vstack([self.M, np.zeros((W.shape[1], M.shape[1]))])
        self.H = np.vstack([np.zeros((self.F.shape[0]+self.W.shape[1], self.M.shape[1])), self.M_stili])

        if np.isscalar(Alpha):
            self.Alpha = Alpha * np.eye(W.shape[1])
        else:
            self.Alpha = Alpha

        self.mu = mu
        self.beta = beta
        self.lambd = self.beta * np.linalg.norm(self.mu * self.M_stili.T @ self.M_stili, ord=2)**2

        self.Q = linalg.block_diag(Q, tol * np.eye(W.shape[1]))
        self.P = linalg.block_diag(np.eye(F.shape[0]), tol * np.eye(W.shape[1]))
        self.R = R

    def compute_matrices(self):

        n = self.F.shape[0]
        l = self.W.shape[1]

        F_aug = np.block([
            [self.F, self.W],
            [np.zeros((l, n)),  self.Alpha]
        ])

        G_aug = np.vstack([self.G, np.zeros((l, self.G.shape[1]))])

        E_F_aug = np.hstack([self.E_F, self.E_W])
        E_G_aug = self.E_G

        bar_I = np.vstack([
            np.eye(n + l),
            np.zeros((E_G_aug.shape[0], n + l))
        ])
        bar_F = np.vstack([F_aug, E_F_aug])
        bar_G = np.vstack([G_aug, E_G_aug])

        Sigma_k = np.block([[self.mu**(-1) * np.eye(self.M_stili.shape[0]) - self.lambd**(-1) * self.M_stili @ self.M_stili.T, np.zeros((self.M_stili.shape[0], self.E_G.shape[0]))],
                            [np.zeros((self.E_G.shape[0], self.M_stili.shape[0])), self.lambd**(-1) * np.eye(self.E_G.shape[0])]])

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
        
        n = self.F.shape[0]
        m = self.G.shape[1]
        l = self.W.shape[1]

        row1 = np.zeros((n + l, n+l))
        row2 = np.zeros((m, n + l))
        row3 = -np.eye(n+l)
        row4 = bar_F
        row5 = np.zeros((n + l, n + l))
        row6 = np.zeros((m, n + l))

        return np.vstack([row1, row2, row3, row4, row5, row6])

    def make_open_Left(self, bar_F):

        n = self.F.shape[0]
        m = self.G.shape[1]
        l = self.W.shape[1]

        row1 = np.hstack([np.zeros((n + l, n+l + m + n+l + bar_F.T.shape[1] )), np.eye(n+l), np.zeros((n+l, m))])
        row2 = np.hstack([np.zeros((m, n+l + m + n+l + bar_F.T.shape[1] + n+l)), np.eye(m)])
        row3 = np.hstack([np.zeros((n+l, n+l + m)), -np.eye(n+l), bar_F.T, np.zeros((n+l, n+l + m))])

        return np.vstack([row1, row2, row3])

    def closed_recursion(self):

        F_aug, G_aug, bar_I, bar_F, bar_G, Sigma_k = self.compute_matrices()
        P_k = self.P

        for _ in range(400):

            inv_P_k = np.linalg.inv(P_k)

            temp = bar_I @ inv_P_k @ bar_I.T + bar_G @ np.linalg.inv(self.R) @ bar_G.T
            temp_inv = np.linalg.inv(temp)

            P_k = self.Q + bar_F.T @ temp_inv @ bar_F

        Lk = inv_P_k @ bar_I.T @ temp_inv @ bar_F 
        Kk = - np.linalg.inv(self.R) @ bar_G.T @ temp_inv @ bar_F 


        return Lk, Kk, P_k

    def open_recursion(self):

        n = self.F.shape[0]
        m = self.G.shape[1]
        l = self.W.shape[1]

        F_aug, G_aug, bar_I, bar_F, bar_G, Sigma_k = self.compute_matrices()
        P_k = self.P

        R_inv = np.linalg.inv(self.R)
        Q_inv = np.linalg.inv(self.Q)

        for _ in range(400):

            Left = self.make_open_Left(bar_F)

            Kernel = self.make_open_Kernel(
                np.linalg.inv(P_k), 
                R_inv, 
                Q_inv, 
                Sigma_k, 
                bar_I, 
                bar_G
            )

            Right = self.make_open_Right(bar_F)

            solution = Left @ np.linalg.solve(Kernel, Right)

            P_k = solution[n+l + m:n+l + m + n+l, :]

        Lk = solution[:n+l, :]
        Kk = solution[n+l:n+l + m, :]

        return Lk, Kk, P_k

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

class MRLQRD(object):
    """Markovian Robust Linear Quadratic Regulator with Disturbances"""
    def __init__(self, F, G, W, E_F, E_G, E_W, M, Alpha, Q, R, \
                 Prob1=np.array([[1]]), Prob2=np.array([[1]]), mu=1e9, beta=1.02, tol = 1e-4):

        self.F = F
        self.G = G
        self.W = W
        self.E_F = E_F
        self.E_G = E_G
        self.E_W = E_W
        self.M = M

        self.Prob1 = Prob1
        self.Prob2 = Prob2

        self.modes1 = Prob1.shape[0]
        self.modes2 = Prob2.shape[0]

        self.M_stili = [np.vstack([self.M[i], np.zeros((W[i].shape[1], M[i].shape[1]))]) for i in range(M.shape[0])]
        self.H = [np.zeros_like((M.shape[0], self.F.shape[0]+self.W.shape[1]+M.shape[1]+W.shape[1])) for i in range(M.shape[0])]

        self.mu = mu
        self.beta = beta
        self.lambd = [self.beta * np.linalg.norm(self.mu * self.M_stili[i].T @ self.M_stili[i], ord=2)**2 for i in range(len(self.M_stili))]

        self.Q = [linalg.block_diag(Q[i], tol * np.eye(W[i].shape[1])) for i in range(len(self.Prob1))]
        self.P = [linalg.block_diag(np.eye(F[0].shape[0]), tol * np.eye(W[i].shape[1])) for i in range(len(self.Prob1))]
        self.R = R

        if np.isscalar(Alpha):
            self.Alpha = Alpha * np.eye(W.shape[1])
        else:
            self.Alpha = Alpha

    def compute_matrices(self):

        n = self.F.shape[1]
        l = self.Alpha.shape[1]

        G_aug = [np.vstack([self.G[i], np.zeros((l, self.G.shape[1]))]) for i in range(self.modes1)]

        E_F_aug = [np.hstack([self.E_F[i], self.E_W[i]]) for i in range(len(self.Prob1))]
        E_G_aug = self.E_G

        bar_I = np.vstack([
            np.eye(n + l),
            np.zeros((E_G_aug[0].shape[0], n + l))
        ])

        bar_G = [np.vstack([G_aug[i], E_G_aug[i]]) for i in range(self.modes1)]

        Sigma_k = [np.block([[self.mu**(-1) * np.eye(self.M_stili[i].shape[0]) - self.lambd[i]**(-1) * self.M_stili[i] @ self.M_stili[i].T, np.zeros((self.M_stili[i].shape[0], self.E_G[i].shape[0]))],
                            [np.zeros((self.E_G[i].shape[0], self.M_stili[i].shape[0])), self.lambd[i]**(-1) * np.eye(self.E_G[i].shape[0])]]) for i in range(self.modes1)]


        F_aug = [np.block([
            [self.F[i], self.W[i]],
            [np.zeros((l, n)),  self.Alpha[0]]
        ]) for i in range(self.modes1)]

        bar_F = [np.vstack([F_aug[i], E_F_aug[i]]) for i in range(self.modes1)]

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
        
        n = self.F[0].shape[0]
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

    def closed_recursion(self):

        F_aug, G_aug, bar_I, bar_F, bar_G, Sigma_k = self.compute_matrices()

        # self.mode matrices self.P
        P_k = self.P.copy()
        Kk = np.array([np.zeros((G_aug[i].shape[1], F_aug[i].shape[1])) for i in range(self.modes1)])
        Lk = np.array([np.zeros((F_aug[i].shape[1], F_aug[i].shape[1])) for i in range(self.modes1)])

        for _ in range(400):

            PSI = np.array([np.zeros((F_aug[0].shape[0], F_aug[0].shape[0])) for _ in range(self.modes1)])
            for i in range(self.modes1):
                for j in range(self.modes1):
                    PSI[i] += float(self.Prob1[i, j]) * P_k[j]

                inv_P_k = np.linalg.inv(PSI[i])

                temp = bar_I @ inv_P_k @ bar_I.T + bar_G[i] @ np.linalg.inv(self.R[i]) @ bar_G[i].T
                temp_inv = np.linalg.inv(temp)

                P_k[i] = self.Q[i] + bar_F[i].T @ temp_inv @ bar_F[i]

                Lk[i] = inv_P_k @ bar_I.T @ temp_inv @ bar_F[i]
                Kk[i] = - np.linalg.inv(self.R[i]) @ bar_G[i].T @ temp_inv @ bar_F[i]

        return Lk, Kk, P_k

    def open_recursion(self):

        n = self.F[0].shape[0]
        m = self.G[0].shape[1]
        l = self.W[0].shape[1]

        F_aug, G_aug, bar_I, bar_F, bar_G, Sigma_k = self.compute_matrices()
        P_k = self.P.copy()
        Kk = np.array([np.zeros((G_aug[i].shape[1], F_aug[i].shape[1])) for i in range(self.modes1)])
        Lk = np.array([np.zeros((F_aug[i].shape[1], F_aug[i].shape[1])) for i in range(self.modes1)])

        R_inv = [np.linalg.inv(self.R[i]) for i in range(self.modes1)]
        Q_inv = [np.linalg.inv(self.Q[i]) for i in range(self.modes1)]

        for _ in range(400):

            PSI = np.array([np.zeros((F_aug[0].shape[0], F_aug[0].shape[0])) for _ in range(self.modes1)])
            for i in range(self.modes1):
                for j in range(self.modes1):
                    PSI[i] += float(self.Prob1[i, j]) * P_k[j]

                # Print Riccati convergence information
                # print(f"Mode {i+1}, Iteration {_+1}, P_k norm: {np.linalg.norm(P_k[i], ord='fro')}")

                Left = self.make_open_Left(bar_F[i])

                Kernel = self.make_open_Kernel(
                    np.linalg.inv(P_k[i]), 
                    R_inv[i], 
                    Q_inv[i], 
                    Sigma_k[i], 
                    bar_I, 
                    bar_G[i]
                )

                Right = self.make_open_Right(bar_F[i])

                solution = Left @ np.linalg.solve(Kernel, Right)

                P_k[i] = solution[n+l + m:n+l + m + n+l, :]

                P_k[i][l,l] = 0

                Lk[i] = solution[:n+l, :]
                Kk[i] = solution[n+l:n+l + m, :]

        return Lk, Kk, P_k

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

if __name__ == "__main__":

    F = np.array([[1.1, 0, 0], 
                   [0, 0, 1.2],
                   [-1, 1, 0]])  # state transition matrix
    G = np.array([[0, 1],
                   [1, 1],
                   [-1, 0]])  # control input matrix
    W = np.array([[0.2],
                   [-0.5],
                   [0]])  # disturbance vector
    M = np.array([[0.7],
                   [0.5],
                   [-0.7]])  # disturbance input matrix

    E_F = np.array([[0.4, 0.5, -0.6]])
    E_G = np.array([[0.4, -0.4]])
    E_W = np.array([[0.5]])

    Alpha = np.array([[[0.99]],
                      [[0.98]]])


    #rlqrd = RLQRD(F, G, W, E_F, E_G, E_W, M, Alpha, Q=np.eye(3), R=np.eye(2), mu=1e9, beta=1.02, tol=1e-4)
    rlqrd = MRLQRD(F, G, W, E_F, E_G, E_W, M, Alpha, Q=np.eye(3), R=np.eye(2), Prob=np.eye(2), mu=1e9, beta=1.02, tol=1e-4)
    matrices = rlqrd.compute_matrices()
    
    print("Closed Solution")
    Lk, Kk, P_k = rlqrd.closed_recursion()
    print("Lk:", Lk)
    print("Kk:", Kk)
    print("P_k:", P_k)

    print("\nOpen Solution")
    Lk, Kk, P_k = rlqrd.open_recursion()
    print("Lk:", Lk)
    print("Kk:", Kk)
    print("P_k:", P_k)
