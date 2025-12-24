import numpy as np
import random
from multiprocessing import Pool
from tqdm import tqdm
from .soap import soap_param_hash

def load_cache(symbols_str, prefix, cache_dir):
    tag = soap_param_hash()
    soap = np.load(f"{cache_dir}/{prefix}SOAP_{tag}_{symbols_str}.npy", mmap_mode="r")
    idx = np.load(f"{cache_dir}/{prefix}INDEX_{symbols_str}.npy")
    return soap, idx

def similarity_dedup(soap, simlT):
    N = soap.shape[0]
    norms = np.linalg.norm(soap, axis=1)
    remaining = list(range(N))
    uniq, test = [], []

    while remaining:
        i = remaining[0]
        ref, ref_norm = soap[i], norms[i]
        similar = [i]
        next_remain = []
        for j in remaining[1:]:
            cos_sim = ref.dot(soap[j]) / (ref_norm * norms[j] + 1e-12)
            if 1 - cos_sim <= simlT:
                similar.append(j)
            else:
                next_remain.append(j)
        uniq.append(similar[0])
        if len(similar) > 2:
            test.append(random.choice(similar[1:]))
        remaining = next_remain
    return uniq, test

def run_deduplication(groups, prefix, cache_dir, simlT, nproc):
    tasks = [(k, prefix, cache_dir, simlT) for k in groups.keys()]
    uniq_all, test_all = [], []

    with Pool(nproc) as pool:
        for symbols_str, uniq_local, test_local in tqdm(
            pool.imap_unordered(_worker, tasks), total=len(tasks), desc=f"Dedup {prefix}"):
            indices = np.array(groups[symbols_str]["indices"]) 
            uniq_all.extend(indices[uniq_local].tolist())
            test_all.extend(indices[test_local].tolist())

    return uniq_all, test_all

def _worker(args):
    symbols_str, prefix, cache_dir, simlT = args
    soap, idx_map = load_cache(symbols_str, prefix, cache_dir)
    uniq_i, test_i = similarity_dedup(soap, simlT)
    return symbols_str, idx_map[uniq_i].tolist(), idx_map[test_i].tolist()