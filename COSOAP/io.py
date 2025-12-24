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
                if f.lower().endswith(('.xyz', '.extxyz', '.traj')):
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

def write_outputs(atoms_all, train_label_idx, train_unlabel_idx, test_label_idx, test_unlabel_idx):
    write("train_labeled.xyz", [atoms_all[i] for i in train_label_idx])
    write("train_unlabeled.xyz", [atoms_all[i] for i in train_unlabel_idx])
    write("test_labeled.xyz", [atoms_all[i] for i in test_label_idx])
    write("test_unlabeled.xyz", [atoms_all[i] for i in test_unlabel_idx])
    print(f"[OUTPUT] train_labeled.xyz: {len(train_label_idx)}, "
            f"train_unlabeled.xyz: {len(train_unlabel_idx)}, "
            f"test_labeled.xyz: {len(test_label_idx)}, "
            f"test_unlabeled.xyz: {len(test_unlabel_idx)}." )
