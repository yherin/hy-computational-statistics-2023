# Write your functions lp1() and lp2() to replace the functions here. Do not change the names of the functions.
# Then copy the content of this cell into a separate file named 'problem2.py' to be submitted separately on Moodle.
# The file should include these import instructions and no others.
# Note that 'problem2.py' should be a standard Python source file (i.e., a text file, not a Jupyter notebook).

import numpy as np
from scipy.special import gammaln, multigammaln, logsumexp
import numpy as np
import scipy.linalg as slg
from scipy.special import multigammaln


def lp1(x: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    """Returns log p(x | v, w).
    Input:
    x: double
    v: np.array, shape: (K,)
    w: np.array, shape: (K,)
    """
    
    # Handle values outside of support
    if np.any(x <= 0):
        raise ValueError("x must be positive")
    if v.shape != w.shape:
        raise ValueError("v and w must have the same shape")
    if not np.allclose(w.sum(), 1):
        raise ValueError("w must sum to 1")
    if np.any(v <= 0):
        raise ValueError("v must be positive")

    term0 = np.log(w)
    term1 = v * np.log(2 + v)
    term2 = gammaln(v)
    term3 = (2*v - 1) * np.log(x)
    term4 = -v*(x**2)
    return logsumexp(
        term0 + term1 - term2 + term3 + term4
    )



def lp2(X: np.ndarray, Psi: np.ndarray, nu: int) -> float:
    """Returns log p(x | \Psi, \nu)
    Input:
    x: np.array, shape: (p,p)
    Psi: np.array, shape: (p,p)
    nu: double, \nu > p-1
    """
    # Dimension of the input matrices
    p = X.shape[0]

    if len(X.shape) > 1:
        if X.shape[0] != X.shape[1]:
            raise ValueError("X must be a square matrix.")

    if nu <= p - 1:
        raise ValueError("'nu' must be greater than p - 1.")

    try:
        L = np.linalg.cholesky(X)

        Y = slg.solve_triangular(L, Psi, lower=True)

        Z = slg.solve_triangular(L.T, Y, lower=False)
    except Exception as e:
        raise ValueError(f"Choleksy decomposition failed for X {X}") from e

    try:
        det_psi = np.linalg.det(Psi)
    except Exception as e:
        raise ValueError(f"np.linalt.det(Psi) value {Psi}") from e

    trace_psi__inv = np.trace(Z)

    term1 = nu / 2 * np.log(det_psi)
    term2 = - (nu * p / 2) * np.log(2)
    term3 = - multigammaln(nu / 2, p) 
    term4 = - ((nu + p + 1) / 2) * np.log(np.linalg.det(X))
    term5 = -(1/2) * trace_psi__inv

    return np.sum([term1, term2, term3, term4, term5])

