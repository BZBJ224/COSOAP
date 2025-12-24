import os
from .config import get_args
from .io import read_input, write_outputs
from .soap import split_structures, build_cache
from .dedup import run_deduplication
from .utils import soap_param_hash

def main():
    args = get_args()
    os.makedirs("soap_cache", exist_ok=True)

    atoms_all = read_input(args.input_path, args.nproc)
    center_elements = args.atoms.split()

    labeled, unlabeled = split_structures(atoms_all, center_elements)

    for prefix, groups in [("LABELED_", labeled), ("UNLABELED_", unlabeled)]:
        if not groups:
            continue
        tag = soap_param_hash(args.rcut)
        need_build = any(not os.path.exists(f"soap_cache/{prefix}SOAP_{tag}_{k}.npy")
                         for k in groups)
        if need_build:
            print(f"[CACHE] Building {prefix.strip('_')} SOAP caches...")
            nproc_use = args.nproc if prefix == "LABELED_" else 1
            for key in sorted(groups.keys()):
                build_cache(groups[key], prefix, "soap_cache", nproc_use,
                            center_elements, args.rcut)

    # 去重
    train_idx, test_idx = run_deduplication(labeled, "LABELED_", "soap_cache", args.simlT, args.nproc)
    unlabeled_idx, _ = run_deduplication(unlabeled, "UNLABELED_", "soap_cache", args.simlT, args.nproc)

    # 写出统计
    with open("uniques.csv", "w") as f:
        for name, groups in [("LABELED", labeled), ("UNLABELED", unlabeled)]:
            for k, g in groups.items():
                f.write(f"{k} ({name}),{len(g['indices'])}\n")

    write_outputs(atoms_all, train_idx, test_idx, unlabeled_idx)

if __name__ == "__main__":
    main()