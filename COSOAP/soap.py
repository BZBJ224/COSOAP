import os
import numpy as np
from dscribe.descriptors import SOAP
from multiprocessing import Pool
from tqdm import tqdm
from .utils import soap_param_hash, get_Kpts

def split_structures(atoms_all, center_elements):
    labeled, unlabeled = {}, {}
    center_set = set(center_elements)

    for idx, atoms in enumerate(atoms_all):
        if get_Kpts(atoms.cell) > 100:
            continue
        key = "".join(sorted(set(atoms.symbols)))
        try:
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            per_atom_e = energy / len(atoms)
            max_force = np.linalg.norm(forces, axis=1).max()
            has_label = (-10 <= per_atom_e <= -1) and (max_force <= 10)
        except:
            has_label = False

        target = labeled if has_label else unlabeled
        if key not in target:
            target[key] = {"symbols": sorted(set(atoms.symbols)), "atoms": [], "indices": []}
        target[key]["atoms"].append(atoms)
        target[key]["indices"].append(idx)

    return labeled, unlabeled

def build_cache(group, prefix, cache_dir, nproc, center_elements, rcut=6.0):
    symbols = group["symbols"]
    atoms_list = group["atoms"]
    indices = group["indices"]
    symbols_str = "".join(sorted(symbols))
    tag = soap_param_hash(rcut)
    soap_file = f"{cache_dir}/{prefix}SOAP_{tag}_{symbols_str}.npy"

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
    np.save(f"{cache_dir}/{prefix}ENERGY_{symbols_str}.npy",
            np.array([a.get_potential_energy() if prefix == "LABELED_" else 0
                      for a in atoms_list], dtype=np.float32))
    np.save(f"{cache_dir}/{prefix}INDEX_{symbols_str}.npy", np.array(indices, dtype=np.int32))