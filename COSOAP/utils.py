import hashlib
import numpy as np

def soap_param_hash(rcut=6.0, nmax=8, lmax=6):
    s = f"r{rcut}_n{nmax}_l{lmax}_inner_periodic"
    return hashlib.md5(s.encode()).hexdigest()[:8]

def get_Kpts(cell, Rk=25):
    """Estimate k-point density"""
    if np.linalg.det(cell) == 0:
        return 1000  # 无效cell
    tmp_0 = np.dot(cell[0], np.cross(cell[1], cell[2]))
    n1 = max(1, int(Rk * np.linalg.norm(np.cross(cell[1], cell[2]) / tmp_0) + 0.5))
    n2 = max(1, int(Rk * np.linalg.norm(np.cross(cell[2], cell[0]) / tmp_0) + 0.5))
    n3 = max(1, int(Rk * np.linalg.norm(np.cross(cell[0], cell[1]) / tmp_0) + 0.5))
    return n1 * n2 * n3