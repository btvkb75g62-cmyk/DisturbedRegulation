**Markovian Robust LQR with Disturbances**

This repository implements a **Markovian Robust Linear Quadratic Regulator with Disturbances** for discrete-time systems with:

* Markovian switching dynamics (two independent chains)
* Parametric uncertainties
* Measurable disturbances
* Augmented state formulation

The approach provides a **Riccati-based solution** for robust optimal control with computational complexity on the order of ( \mathcal{O}(n^3) ).

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
from mrlqrd import MRLQRD

F = [np.eye(3), np.eye(3)]
G = [np.ones((3,2)), np.ones((3,2))]
W = [np.ones((3,1)), np.zeros((3,1))]

E_F = [np.ones((1,3)), np.ones((1,3))]
E_G = [np.ones((1,2)), np.ones((1,2))]
E_W = [np.ones((1,1)), np.ones((1,1))]

M = [np.ones((3,1)), np.ones((3,1))]

Prob1 = np.array([[0.5, 0.5],
                  [0.5, 0.5]])

Prob2 = np.array([[0.8, 0.2],
                  [0.8, 0.2]])

controller = MRLQRD(
    F, G, W, E_F, E_G, E_W, M,
    Alpha=[np.array([[0.95]]), np.array([[1.10]])],
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

