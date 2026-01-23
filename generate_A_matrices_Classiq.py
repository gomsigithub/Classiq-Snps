import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import identity, csr_matrix, kron, diags, eye
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import MCXGate, RXGate, CRXGate, QFTGate, RYGate, UCRXGate, UCRYGate, UCRZGate
from qiskit.quantum_info import SparsePauliOp, Statevector, Operator
from qiskit import transpile

def forward_diff_rect(N, dx=1):
    D = np.zeros((N, N))
    for i in range(N-1):
        D[i, i] = -1.0
        D[i, i+1] = 1.0
    return D / dx

def backward_diff_rect(N, dx=1):
    D = np.zeros((N, N))
    for i in range(1, N-1):
        D[i, i] = 1.0
        D[i, i-1] = -1.0
    D[0,0] = 2
    D[N-1, N-2] = -2
    return D / dx 

# Re-defining the fdtd_generator functionality for verification
def GetEzDofNum(i, j, nx, ny):
    if i < 0 or j < 0 or i >= nx or j >= ny:
        return -1
    return j * nx + i

def GetHxDofNum(i, j, nx, ny):
    if i < 0 or j < 0 or i >= nx or j >= ny - 1:
        return -1
    return j * nx + i

def GetHyDofNum(i, j, nx, ny):
    if i < 0 or j < 0 or i >= nx - 1 or j >= ny:
        return -1
    return j * (nx - 1) + i + nx * (ny - 1)

def generate_fdtd_matrices(nx, ny, h, eps_r=1.0):
    eps = 1 * eps_r
    mu0 = 1

    n_e = nx * ny
    n_hx = nx * (ny - 1)
    n_hy = (nx - 1) * ny
    n_h = n_hx + n_hy

    E_coef = 1.0 / (h * eps)
    H_coef = 1.0 / (h * mu0)

    AE_mat = np.zeros((n_e, n_h))
    AH_mat = np.zeros((n_h, n_e))

    # Build AE_mat
    for j in range(ny):
        for i in range(nx):
            ez_ix = GetEzDofNum(i, j, nx, ny)
            hy_plus = GetHyDofNum(i, j, nx, ny)
            hy_minus = GetHyDofNum(i-1, j, nx, ny)
            if hy_plus == -1:
                AE_mat[ez_ix, hy_minus] = -2.0 * E_coef
            elif hy_minus == -1:
                AE_mat[ez_ix, hy_plus] = 2.0 * E_coef
            else:
                AE_mat[ez_ix, hy_plus] = E_coef
                AE_mat[ez_ix, hy_minus] = -E_coef

            hx_plus = GetHxDofNum(i, j, nx, ny)
            hx_minus = GetHxDofNum(i, j-1, nx, ny)
            if hx_plus == -1:
                AE_mat[ez_ix, hx_minus] = 2.0 * E_coef
            elif hx_minus == -1:
                AE_mat[ez_ix, hx_plus] = -2.0 * E_coef
            else:
                AE_mat[ez_ix, hx_plus] = -E_coef
                AE_mat[ez_ix, hx_minus] = E_coef

    # Build AH_mat for Hx
    for j in range(ny - 1):
        for i in range(nx):
            hx_ix = GetHxDofNum(i, j, nx, ny)
            ez_plus = GetEzDofNum(i, j+1, nx, ny)
            ez_minus = GetEzDofNum(i, j, nx, ny)
            AH_mat[hx_ix, ez_plus] = -H_coef
            AH_mat[hx_ix, ez_minus] = H_coef

    # Build AH_mat for Hy
    for j in range(ny):
        for i in range(nx - 1):
            hy_ix = GetHyDofNum(i, j, nx, ny)
            ez_plus = GetEzDofNum(i+1, j, nx, ny)
            ez_minus = GetEzDofNum(i, j, nx, ny)
            AH_mat[hy_ix, ez_plus] = H_coef
            AH_mat[hy_ix, ez_minus] = -H_coef

    return AE_mat, AH_mat

def pad_mat_to_power_of_two(M):
    """
    Pad the matrix M with zeros so that the result
    has shape (2^k, 2^k), where 2^k >= max(M.shape).
    """
    nrow = M.shape[0]
    ncol = M.shape[1]
    krow = int(np.ceil(np.log2(nrow)))
    kcol = int(np.ceil(np.log2(ncol)))

    new_size = max(2**krow, 2**kcol)

    padded = np.zeros((new_size, new_size), dtype=M.dtype)
    padded[:nrow, :ncol] = M
    return padded

def ebu_classical_structured(nx , T , initial_state):

    forward = forward_diff_rect(nx)
    backward = backward_diff_rect(nx)
    sigma00 = [(1, 0), (0, 0)]
    sigma01 = [(0, 1), (0, 0)]
    sigma10 = [(0, 0), (1, 0)]
    sigma11 = [(0, 0), (0, 1)]
    dHydx = np.kron(np.kron(np.kron(sigma01,sigma01),np.eye(nx)),backward)
    dHxdy = np.kron(np.kron(np.kron(sigma01,sigma00),-backward),np.eye(nx))
    dEzdx = np.kron(np.kron(np.kron(sigma10,sigma10),np.eye(nx)),forward)
    dEzdy = np.kron(np.kron(np.kron(sigma10,sigma00),-forward),np.eye(nx))

    A = dHydx + dHxdy + dEzdx + dEzdy

    A1, A2 = hermitian_decomposition(A)
    H = 1j *(A2)
    expAt = expm(A * T)
    EzHxHy = np.matmul(expAt,  initial_state)
    
    return EzHxHy, A, A1, A2

def ebu_classical_unstructured(nx , T , initial_state):

    AE, AH = generate_fdtd_matrices(nx, nx, 1)
    A = np.zeros((nx*nx+2*nx*(nx-1), nx*nx+2*nx*(nx-1)))

    # Place A in top-right
    A[:nx*nx, nx*nx:] = AE

    # Place B in bottom-left
    A[nx*nx:, :nx*nx] = AH

    expAt = expm(A * T)
    EzHxHy = np.matmul(expAt,  initial_state)

    return EzHxHy, A

def hermitian_decomposition(A):
    """
    Decomposes a complex matrix A into Hermitian (A1) and anti-Hermitian (A2) parts
    such that A = A1 + i*A2

    Returns:
    - A1: Hermitian part
    - A2: Hermitian matrix such that i*A2 is anti-Hermitian
    """
    A_dag = A.conj().T
    A1 = 0.5 * (A + A_dag)
    A2 = (-0.5j) * (A - A_dag)
    return A1, A2