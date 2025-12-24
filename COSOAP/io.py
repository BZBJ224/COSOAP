import os
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from ase.io import read, write

def read_single_file(filename):
    try:
        return read(filename, index=":", parallel=False)
    except Exception as e:
        print(f"[ERROR] Reading {filename}: {e}")
        return []

def read_input(input_path, nproc):
    input_path = input_path.strip()
    if os.path.isfile(input_path):
        print(f"[IO] Reading single file: {input_path}")
        return read(input_path, ":")
    elif os.path.isdir(input_path):
        print(f"[IO] Reading folder: {input_path}")
        xyz_files = []
        for root, _, files in os.walk(input_path):
            for f in files:
                if f.lower().endswith(('.xyz', '.extxyz')):
                    xyz_files.append(os.path.join(root, f))
        if not xyz_files:
            raise ValueError("No .xyz files found in folder")
        xyz_files.sort()
        print(f"[IO] Found {len(xyz_files)} files, reading with {nproc} processes...")
        with Pool(nproc) as pool:
            results = list(tqdm(pool.imap_unordered(read_single_file, xyz_files),
                                total=len(xyz_files), desc="Reading files"))
        atoms = [a for sub in results for a in sub]
        print(f"[IO] Total structures loaded: {len(atoms)}")
        return atoms
    else:
        raise ValueError(f"Input path not found: {input_path}")

def write_outputs(atoms_all, train_idx, test_idx, unlabeled_idx):
    write("train.xyz", [atoms_all[i] for i in train_idx])
    write("test.xyz", [atoms_all[i] for i in test_idx])
    write("unlabeled.xyz", [atoms_all[i] for i in unlabeled_idx])
    print(f"[OUTPUT] train.xyz: {len(train_idx)}, test.xyz: {len(test_idx)}, "
          f"unlabeled.xyz: {len(unlabeled_idx)}")