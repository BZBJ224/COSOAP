import os
import numpy as np
from dscribe.descriptors import SOAP
from multiprocessing import Pool
from tqdm import tqdm
from .utils import soap_param_hash, get_Kpts

# soap.py
def split_structures(atoms_all, center_elements):
    """
    按化学组成分组，所有结构（无论有无标签）放在一起
    同时记录每个结构是否有标签
    """
    groups = {}
    center_set = set(center_elements)

    for idx, atoms in enumerate(atoms_all):
        if get_Kpts(atoms.cell) > 100:
            continue

        key = "".join(sorted(atoms.symbols))

        # 判断是否有标签
        try:
            if (-10 <= per_atom_e <= -1) and (max_force <= 10):
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                per_atom_e = energy / len(atoms)
                max_force = np.linalg.norm(forces, axis=1).max()
                has_label = True
            else: break
        except:
            has_label = False

        if key not in groups:
            groups[key] = {
                "symbols": sorted(atoms.symbols),
                "atoms": [],
                "indices": [],
                "has_label": []   # 新增：记录每个结构是否有标签
            }

        groups[key]["atoms"].append(atoms)
        groups[key]["indices"].append(idx)
        groups[key]["has_label"].append(has_label)

    return groups  

def build_cache(group, cache_dir, nproc, center_elements, rcut=6.0):
    symbols = group["symbols"]
    atoms_list = group["atoms"]
    #indices = group["indices"]
    indices = range(len(atoms_list))
    symbols_str = "".join(sorted(symbols))
    tag = soap_param_hash(rcut)
    soap_file = f"{cache_dir}/SOAP_{tag}_{symbols_str}.npy"

    if os.path.exists(soap_file):
        return

    soap = SOAP(species=symbols, r_cut=rcut, n_max=8, l_max=6,
                average="inner", periodic=True)

    centers_list = [[i for i, a in enumerate(atoms) if a.symbol in center_elements]
                    for atoms in atoms_list]

    N = len(atoms_list)
    descs = []

    if N < nproc * 4:
        for atoms, centers in zip(atoms_list, centers_list):
            descs.append(soap.create(atoms, centers))
    else:
        chunk_size = max(1, N // nproc)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_atoms = atoms_list[start:end]
            chunk_centers = centers_list[start:end]
            with Pool(nproc) as p:
                chunk_descs = p.starmap(soap.create, zip(chunk_atoms, chunk_centers))
            descs.extend(chunk_descs)

    np.save(soap_file, np.array(descs, dtype=np.float32))
    np.save(f"{cache_dir}/INDEX_{symbols_str}.npy", np.array(indices, dtype=np.int32))

    np.save(f"{cache_dir}/HAS_LABEL_{symbols_str}.npy", np.array(group["has_label"], dtype=bool))

