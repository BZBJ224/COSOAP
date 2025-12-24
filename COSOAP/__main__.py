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

    groups = split_structures(atoms_all, center_elements)  # 只有一个 groups


    tag = soap_param_hash(args.rcut)
    need_build = any(not os.path.exists(f"soap_cache/SOAP_{tag}_{k}.npy")
                         for k in groups)
    if need_build:
        print(f"[CACHE] Building SOAP caches...")
        for key in sorted(groups.keys()):
            build_cache(groups[key], cache_dir="soap_cache", nproc=args.nproc,
                        center_elements=center_elements, rcut=args.rcut)

    train_label_idx, train_unlabel_idx, test_label_idx, test_unlabel_idx = \
                run_deduplication(groups, cache_dir="soap_cache", simlT=args.simlT, nproc=args.nproc)
    all_used = set(train_label_idx + train_unlabel_idx + test_label_idx + test_unlabel_idx)

    write_outputs(atoms_all, train_label_idx, train_unlabel_idx, test_label_idx, test_unlabel_idx)


if __name__ == "__main__":
    main()
