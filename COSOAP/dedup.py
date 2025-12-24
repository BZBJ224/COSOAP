import numpy as np
import random
from multiprocessing import Pool
from tqdm import tqdm
from .soap import soap_param_hash

def load_cache_with_label(symbols_str, cache_dir):
    tag = soap_param_hash()
    soap = np.load(f"{cache_dir}/SOAP_{tag}_{symbols_str}.npy", mmap_mode="r")
    idx = np.load(f"{cache_dir}/INDEX_{symbols_str}.npy")
    has_label = np.load(f"{cache_dir}/HAS_LABEL_{symbols_str}.npy")
    return soap, idx, has_label

def similarity_dedup(soap, has_label_list, simlT):
    """
    全局去重，但优先保留有标签的结构
    """
    N = soap.shape[0]
    norms = np.linalg.norm(soap, axis=1)
    has_label = np.array(has_label_list)

    remaining = list(range(N))
    uniq_label_idx, uniq_unlabel_idx = [], []
    test_label_idx, test_unlabel_idx = [], []

    while remaining:
        i = remaining[0]
        ref, ref_norm = soap[i], norms[i]

        similar_group = [i]
        next_remain = []

        for j in remaining[1:]:
            cos_sim = ref.dot(soap[j]) / (ref_norm * norms[j] + 1e-12)
            if 1 - cos_sim <= simlT:
                similar_group.append(j)
            else:
                next_remain.append(j)

        # === 关键：优先选择有标签的作为代表 ===
        if len(similar_group) >= 2:
            labeled_in_group = [k for k in similar_group if has_label[k]]
            if labeled_in_group:
                uniq_label_idx.append(labeled_in_group[0])  # 选第一个有标签的
                if len(labeled_in_group) >= 2:
                    test_label_idx.append(labeled_in_group[-1])
                else:
                    test_unlabel_idx.append([k for k in similar_group if k!= labeled_in_group[0]][0])
            else:
                uniq_unlabel_idx.append(similar_group[0])
                if len(labeled_in_group) >= 2:
                    test_unlabel_idx.append(similar_group[1])
        else:
            if has_label[similar_group[0]]: uniq_label_idx.append(similar_group[0])
            else: uniq_unlabel_idx.append(similar_group[0])

        remaining = next_remain

    return uniq_label_idx, uniq_unlabel_idx, test_label_idx, test_unlabel_idx


def run_deduplication(groups, cache_dir, simlT, nproc):
    tasks = [(k,  cache_dir, simlT) for k in groups.keys()]
    uniq_label_all, uniq_unlabel_all, test_label_all, test_unlabel_all = [], [], [], []

    with Pool(nproc) as pool:
        for symbols_str, uniq_label_idx, uniq_unlabel_idx, test_label_idx, test_unlabel_idx in tqdm(
            pool.imap_unordered(_worker, tasks), total=len(tasks), desc=f"Dedup"):
            indices = np.array(groups[symbols_str]["indices"]) 
            uniq_label_all.extend(indices[uniq_label_idx].tolist())
            uniq_unlabel_all.extend(indices[uniq_unlabel_idx].tolist())
            test_label_idx.extend(indices[test_label_idx].tolist())
            test_unlabel_idx.extend(indices[test_unlabel_idx].tolist())

    return uniq_label_all, uniq_unlabel_all, test_label_all, test_unlabel_all

def _worker(args):
    symbols_str, cache_dir, simlT = args
    soap, idx_map, has_label = load_cache_with_label(symbols_str, cache_dir)
    uniq_label_idx, uniq_unlabel_idx, test_label_idx, test_unlabel_idx = similarity_dedup(soap, has_label, simlT)
    return symbols_str, idx_map[uniq_label_idx].tolist(), idx_map[uniq_unlabel_idx].tolist(), \
            idx_map[test_label_idx].tolist(), idx_map[test_unlabel_idx].tolist()
