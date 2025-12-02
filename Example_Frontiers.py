import numpy as np
from RLQRD import RLQRD, MRLQRD
from scipy import linalg
import pandas as pd

"""
This examples was presented in the Frontiers in Control Engineering paper:

Flüs, Patrick, and Olaf Stursberg. 
"Control of Jump Markov Uncertain Linear Systems With General Probability Distributions." 
Frontiers in Control Engineering 3 (2022): 806543. 

@article{flus2022control,
  title={Control of Jump Markov Uncertain Linear Systems With General Probability Distributions},
  author={Fl{\"u}s, Patrick and Stursberg, Olaf},
  journal={Frontiers in Control Engineering},
  volume={3},
  pages={806543},
  year={2022},
  publisher={Frontiers Media SA}
}
"""

A1 = np.array([[1.0, 0.1], 
               [-0.2, 0.9]])

A2 = np.array([[-0.9, -0.1], 
               [0.0, 1.1]])

A3 = np.array([[0.8, 0.0], 
               [0.05, 0.9]])

A4 = np.array([[0.8, 0.0], 
               [-0.2, 0.9]])

A = [A1, A2, A3, A4]
A = np.array(A)

EA1 = np.zeros((2,2))
EA2 = np.zeros((2,2))
EA3 = np.zeros((2,2))
EA4 = np.zeros((2,2))

EA = [EA1, EA2, EA3, EA4]
EA = np.array(EA)

B1 = np.array([[-1.0, 0.0],
               [0.5, 1.0]])

B2 = np.array([[1.0, 0.5],
               [0.0, 2.0]])

B3 = np.array([[0.5, 0.2],
               [0.0, 1.0]])

B4 = np.array([[-1.0, 1.0],
               [0.5, 0.8]])

B = [B1, B2, B3, B4]
B = np.array(B)

EB1 = np.array([[1, 1],
                [1, 1]])/4
EB2 = np.array([[1, 1],
                [1, 1]])/4
EB3 = np.array([[1, 1],
                [1, 1]])/4
EB4 = np.array([[1, 1],
                [1, 1]])/4

EB = [EB1, EB2, EB3, EB4]
EB = np.array(EB)*0

W1 = np.zeros((2,2)) + np.eye(2)
W2 = np.zeros((2,2)) + np.eye(2)
W3 = np.zeros((2,2)) + np.eye(2)
W4 = np.zeros((2,2)) + np.eye(2)

W = [W1, W2, W3, W4]
W = np.array(W)

EW1 = np.array([[1.3, 0.0],
               [0.0, 1.0]])

EW2 = np.array([[0.7, -0.4],
               [0.0, 1.1]])

EW3 = np.array([[0.8, 0.0],
               [0.0, 0.7]])

EW4 = np.array([[0.8, -0.2],
               [0.5, 0.3]])

EW = [EW1, EW2, EW3, EW4]
EW = np.array(EW)
W = np.array(EW)

Prob1 = np.array([[0.25, 0.25, 0.25, 0.25],
                 [0.25, 0.25, 0.25, 0.25],
                 [0.25, 0.25, 0.25, 0.25],
                 [0.25, 0.25, 0.25, 0.25]])

Prob2 = np.array([[1.0]])

Q = np.array([[1.0, 0.0],
               [0.0, 1.0]])
Q = np.array([Q, Q, Q, Q])*1000000

#R = 20 * np.eye(2)
R = 0.0001 * np.eye(2)
R = np.array([R, R, R, R])

Alpha = np.array([np.eye(2)])

M = np.eye(2)
M = np.array([M, M, M, M])

#  F, G, W, E_F, E_G, E_W, M, Alpha, Q, R, Prob=np.array([[1]])

Markovian_regulator = MRLQRD(A, B, W, EA, EB, EW, M, Alpha, Q, R, Prob1, Prob2)

L, K, P = Markovian_regulator.main()

# f0 = np.vstack((EW1, np.zeros((2,2))))
# f00 = np.vstack((EB1, B1))
# f0 = - np.linalg.inv(f00.T @ f00) @ f00.T @ f0

# f1 = np.vstack((EW2, np.zeros((2,2))))
# f11 = np.vstack((EB2, B2))
# f1 = - np.linalg.inv(f11.T @ f11) @ f11.T @ f1

# f2 = np.vstack((EW3, np.zeros((2,2))))
# f22 = np.vstack((EB3, B3))
# f2 = - np.linalg.inv(f22.T @ f22) @ f22.T @ f2

# f3 = np.vstack((EW4, np.zeros((2,2))))
# f33 = np.vstack((EB4, B4))
# f3 = - np.linalg.inv(f33.T @ f33) @ f33.T @ f3

# K[0] = np.hstack((K[0][:,:2], f0))
# K[1] = np.hstack((K[1][:,:2], f1))
# K[2] = np.hstack((K[2][:,:2], f2))
# K[3] = np.hstack((K[3][:,:2], f3))

print("Gains K_i:")
for i in range(len(K)):
    print(f"K_{i+1} = \n{K[i]}\n")

for i in range(len(K)):
    print(f"||E_F + E_G*K_{i+1}|| = \n{np.linalg.norm(EA[i] + EB[i] @ K[i][:,:2], ord='fro')}\n")
    print(f"||E_W + E_G*f_{i+1}|| = \n{np.linalg.norm(EW[i] + EB[i] @ K[i][:,2:], ord='fro')}\n")
    print(f"||W + G*f_{i+1}|| = \n{np.linalg.norm(W[i] + B[i] @ K[i][:,2:], ord='fro')}\n")
## Simulation

def dynamics(x, u, w, mode, A, B, W, EA, EB, EW):
    n = A[0].shape[0]
    m = B[0].shape[1]
    l = W[0].shape[1]
    
    A_nom = A[mode]
    B_nom = B[mode]
    W_nom = W[mode]
    
    delta = np.random.uniform(-1, 1)

    A_err = delta * EA[mode]
    B_err = delta * EB[mode]
    W_err = delta * EW[mode]

    # Here is wrong
    x_next = (A_nom + A_err) @ x + (B_nom + B_err) @ u + (W_nom + W_err) @ w
    
    return x_next

Sim_time = 500
x0 = np.array([[1.0],
               [0.0]])
modes = [0, 1, 2, 3]
mode_names = ['1', '2', '3', '4']
mode = 0

X = np.zeros((2, Sim_time))
U = np.zeros((2, Sim_time))
Mode = np.zeros((1, Sim_time))  

w = np.array([[1],
              [1]])

x = x0

for t in range(Sim_time):

    mode = np.random.choice(modes, p=Prob1[mode])
    X[:, t] = x.flatten()
    Mode[:, t] = mode + 1
    u = K[mode] @ np.vstack((x, w))
    U[:, t] = u.flatten()
    x = dynamics(x, u, w, mode, A, B, W, EA, EB, EW)

# Plot State x1 vs x2
# Plot Control u1 vs u2
# Plot Mode vs Time
import matplotlib.pyplot as plt
fig, axs = plt.subplots(3, 1, figsize=(8, 12))
axs[0].plot(range(Sim_time), X[0, :], label='x1')
axs[0].plot(range(Sim_time), X[1, :], label='x2')
axs[0].set_title('State Trajectories')
axs[0].set_xlabel('Time Step')
axs[0].set_ylabel('State Value')
axs[0].legend()
axs[0].grid()
axs[1].plot(range(Sim_time), U[0, :], label='u1')
axs[1].plot(range(Sim_time), U[1, :], label='u2')
axs[1].set_title('Control Inputs')
axs[1].set_xlabel('Time Step')
axs[1].set_ylabel('Control Value')
axs[1].legend()
axs[1].grid()
axs[2].step(range(Sim_time), Mode[0, :], where='post')
axs[2].set_title('Mode Trajectory')
axs[2].set_xlabel('Time Step')
axs[2].set_ylabel('Mode')       
axs[2].set_yticks([1, 2, 3, 4])
axs[2].set_yticklabels(mode_names)
axs[2].grid()
plt.tight_layout()
plt.show()
