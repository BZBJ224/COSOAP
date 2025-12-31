import os
import numpy as np
from .config import get_args
from .io import read_input, write_outputs
from .soap import split_structures, build_cache
from .dedup import run_deduplication
from .utils import soap_param_hash

def main():
    args = get_args()
    os.makedirs("soap_cache", exist_ok=True)

    # 1. 读取
    atoms_all = read_input(args.input_path, args.nproc)
    
    # 2. 分组
    center_elements = args.atoms.split()
    groups = split_structures(atoms_all, center_elements)
    
    # 3. 准备参数 / 分配数量
    dedup_param = None # 传递给 dedup 的主参数
    
    if args.mode == "fps":
        # === FPS 模式：执行分配算法 (修复版) ===
        valid_total_structures = sum(len(g["atoms"]) for g in groups.values())
        target_total = min(args.num, valid_total_structures)
        
        print(f"[FPS] Distributing {target_total} samples from {valid_total_structures} structures...")
        
        allocations = {}
        remainders = []
        current_sum = 0
        
        for key in groups:
            n_this_group = len(groups[key]["atoms"])
            if valid_total_structures > 0:
                raw_share = target_total * (n_this_group / valid_total_structures)
            else:
                raw_share = 0
            base_alloc = int(raw_share)
            allocations[key] = base_alloc
            current_sum += base_alloc
            
            if base_alloc < n_this_group:
                remainders.append((raw_share - base_alloc, key))
            else:
                remainders.append((-1.0, key)) # 已满

        deficit = target_total - current_sum
        remainders.sort(key=lambda x: x[0], reverse=True)
        
        for i in range(deficit):
            if i >= len(remainders): break
            key_to_add = remainders[i][1]
            if allocations[key_to_add] < len(groups[key_to_add]["atoms"]):
                allocations[key_to_add] += 1

        for key in groups:
            groups[key]["n_select"] = allocations[key]
            print(f"  - Group {key:<10}: Select {allocations[key]}")
            
        dedup_param = None # FPS 模式下，参数已经写入 groups 字典里了
        
    else:
        # === Threshold 模式：无需分配 ===
        print(f"[Threshold] Running Cosine Deduplication with simlT = {args.simlT}")
        dedup_param = args.simlT

    # 4. 缓存
    tag = soap_param_hash(args.rcut)
    need_build = any(not os.path.exists(f"soap_cache/SOAP_{tag}_{k}.npy") for k in groups)
    if need_build:
        print(f"[CACHE] Building SOAP caches...")
        for key in sorted(groups.keys()):
            build_cache(groups[key], cache_dir="soap_cache", nproc=args.nproc,
                        center_elements=center_elements, rcut=args.rcut)

    # 5. 运行筛选
    # 将 mode 和 param 传进去
    train_label_idx, train_unlabel_idx, test_label_idx, test_unlabel_idx = \
                run_deduplication(groups, cache_dir="soap_cache", nproc=args.nproc, 
                                  mode=args.mode, param=dedup_param)

    # 6. 输出
    write_outputs(atoms_all, train_label_idx, train_unlabel_idx, test_label_idx, test_unlabel_idx)

if __name__ == "__main__":
    main()
