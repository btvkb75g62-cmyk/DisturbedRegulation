**Markovian Robust LQR with Disturbances**

This repository implements a **Markovian Robust Linear Quadratic Regulator with Disturbances** for discrete-time systems with:

* Markovian switching dynamics (two independent chains)
* Parametric uncertainties
* Measurable disturbances

---

* Supports **two Markov chains**:
  * System modes (`Prob1`)
  * Disturbance modes (`Prob2`)
* Handles **parametric uncertainty** via structured matrices ( E_F, E_G, E_W )

---

## Usage

Basic example:

```python
import numpy as np

F = [np.array([[1.1, 0, 0], 
              [0, 0, 1.2],
              [-1, 1, 0]]),np.array([[-0.8, 0, 0], 
                                     [0, 1, 0.5],
                                     [1.3, 0, -1]])]  

G = [np.array([[0, 1],
              [1, 1],
              [-1, 0]]), np.array([[0.1, -1],
                                   [1, 3],
                                   [0.5, 0.3]])]  

W = [np.array([[0.2],
              [-0.5],
              [0.7]]), np.array([[0.2],
                                 [-1],
                                 [1.2]])]  

M = [np.array([[0.7],
              [0.5],
              [-0.7]]), np.array([[0.7],
                                  [0.5],
                                  [-0.7]])]

E_F = [np.array([[0.4, 0.5, -0.6]]), np.array([[0.0, -1.0, 0.0]])]
E_G = [np.array([[0.4, -0.4]]), np.array([[-0.8, 0.5]])]
E_W = [np.array([[-0.70]]), np.array([[1.6]])]

Lambda = [np.array([[0.95]]), 
                            np.array([[1.10]])]

Prob1=np.array([[0.5, 0.5], 
               [0.5, 0.5]])
Prob2=np.array([[0.8, 0.2], 
               [0.8, 0.2]])

controller = MRLQRD(
    F, G, W, E_F, E_G, E_W, M, Alpha,
    Q=np.eye(3),
    R=np.eye(2),
    Prob1=Prob1,
    Prob2=Prob2
)

L, K, P = controller.main()
```
---

## License

MIT License

---

## Author

Carlos A. F. Persiani

* [carlospersiani@usp.br](mailto:carlospersiani@usp.br)
* [carlospersiani@gmail.com](mailto:carlospersiani@gmail.com) 

