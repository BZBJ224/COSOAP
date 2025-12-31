import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from .soap import soap_param_hash

def load_cache_with_label(symbols_str, cache_dir):
    tag = soap_param_hash()
    soap = np.load(f"{cache_dir}/SOAP_{tag}_{symbols_str}.npy", mmap_mode="r")
    idx = np.load(f"{cache_dir}/INDEX_{symbols_str}.npy")
    has_label = np.load(f"{cache_dir}/HAS_LABEL_{symbols_str}.npy")
    return soap, idx, has_label

# =========================================================
# 算法 1: FPS (最远点采样) - 指定数量
# =========================================================
def fps_selection_target_n(soap, has_label_list, n_select):
    """
    基于数量的 FPS 筛选，使用欧几里得距离 (对归一化SOAP等价于Cosine距离)
    """
    N = soap.shape[0]
    has_label = np.array(has_label_list, dtype=bool)
    
    uniq_label_idx, uniq_unlabel_idx = [], []
    all_indices = np.arange(N)
    labeled_indices = all_indices[has_label]
    unlabeled_indices = all_indices[~has_label]
    
    n_labeled = len(labeled_indices)
    min_dists = np.full(N, np.inf, dtype=np.float32)
    current_count = 0
    
    # --- Phase 1: Labeled ---
    n_from_labeled = min(n_labeled, n_select)
    if n_from_labeled > 0:
        first_idx = labeled_indices[0]
        uniq_label_idx.append(first_idx)
        dists = np.linalg.norm(soap - soap[first_idx], axis=1)
        min_dists = np.minimum(min_dists, dists)
        current_count += 1
        
        remaining_mask = np.ones(n_labeled, dtype=bool)
        remaining_mask[0] = False
        
        for _ in range(n_from_labeled - 1):
            current_dists_subset = min_dists[labeled_indices]
            masked_dists = current_dists_subset.copy()
            masked_dists[~remaining_mask] = -1.0
            
            best_local_idx = np.argmax(masked_dists)
            best_global_idx = labeled_indices[best_local_idx]
            
            uniq_label_idx.append(best_global_idx)
            remaining_mask[best_local_idx] = False
            current_count += 1
            
            new_dists = np.linalg.norm(soap - soap[best_global_idx], axis=1)
            min_dists = np.minimum(min_dists, new_dists)

    # --- Phase 2: Unlabeled ---
    n_needed = n_select - current_count
    if n_needed > 0 and len(unlabeled_indices) > 0:
        remaining_mask = np.ones(len(unlabeled_indices), dtype=bool)
        
        if current_count == 0:
            first_idx = unlabeled_indices[0]
            uniq_unlabel_idx.append(first_idx)
            dists = np.linalg.norm(soap - soap[first_idx], axis=1)
            min_dists = np.minimum(min_dists, dists)
            remaining_mask[0] = False
            n_needed -= 1

        for _ in range(n_needed):
            if not np.any(remaining_mask): break
            current_dists_subset = min_dists[unlabeled_indices]
            masked_dists = current_dists_subset.copy()
            masked_dists[~remaining_mask] = -1.0
            
            best_local_idx = np.argmax(masked_dists)
            best_global_idx = unlabeled_indices[best_local_idx]
            
            uniq_unlabel_idx.append(best_global_idx)
            remaining_mask[best_local_idx] = False
            
            new_dists = np.linalg.norm(soap - soap[best_global_idx], axis=1)
            min_dists = np.minimum(min_dists, new_dists)

    # 剩余的归为 Test
    selected_set = set(uniq_label_idx + uniq_unlabel_idx)
    test_label_idx = [i for i in labeled_indices if i not in selected_set]
    test_unlabel_idx = [i for i in unlabeled_indices if i not in selected_set]

    return uniq_label_idx, uniq_unlabel_idx, test_label_idx, test_unlabel_idx

# =========================================================
# 算法 2: Cosine Threshold (贪婪去重) - 指定阈值
# =========================================================
def cosine_threshold_dedup(soap, has_label_list, simlT):
    """
    原始的贪婪去重逻辑：相似度 > (1-simlT) 则丢弃
    """
    N = soap.shape[0]
    # 需要计算 Norms 进行 Cosine 计算
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

        # 遍历剩余点
        for j in remaining[1:]:
            # Cosine Sim
            cos_sim = ref.dot(soap[j]) / (ref_norm * norms[j] + 1e-12)
            # 如果相似 (1 - cos <= simlT)
            if 1 - cos_sim <= simlT:
                similar_group.append(j)
            else:
                next_remain.append(j)

        # 组内优先选有标签
        if len(similar_group) >= 2:
            labeled_in_group = [k for k in similar_group if has_label[k]]
            if labeled_in_group:
                uniq_label_idx.append(labeled_in_group[0])
                if len(labeled_in_group) >= 2:
                    test_label_idx.append(labeled_in_group[-1])
                else:
                    # 选一个无标签的做测试
                    others = [k for k in similar_group if k!= labeled_in_group[0]]
                    if others: test_unlabel_idx.append(others[0])
            else:
                uniq_unlabel_idx.append(similar_group[0])
                if len(similar_group) >= 2:
                    test_unlabel_idx.append(similar_group[1])
        else:
            if has_label[similar_group[0]]: uniq_label_idx.append(similar_group[0])
            else: uniq_unlabel_idx.append(similar_group[0])

        remaining = next_remain

    return uniq_label_idx, uniq_unlabel_idx, test_label_idx, test_unlabel_idx


def run_deduplication(groups, cache_dir, nproc, mode, param):
    """
    mode: 'fps' or 'threshold'
    param: n_select (if fps) OR simlT (if threshold)
    注意：如果 mode='fps'，param 应该是一个字典 {key: n_select} 或者在 groups 里面读取
    """
    tasks = []
    for k in groups.keys():
        if mode == 'fps':
            # 从 groups 字典中获取分配好的数量
            p_val = groups[k].get("n_select", 0)
        else:
            # 直接使用传入的阈值 simlT
            p_val = param
            
        tasks.append((k, cache_dir, mode, p_val))
        
    uniq_label_all, uniq_unlabel_all = [], []
    test_label_all, test_unlabel_all = [], []

    desc_str = "FPS Selection" if mode == "fps" else "Cosine Dedup"

    with Pool(nproc) as pool:
        for _, u_lbl, u_unlbl, t_lbl, t_unlbl in tqdm(
            pool.imap_unordered(_worker, tasks), total=len(tasks), desc=desc_str):
            
            # _worker 返回的是局部索引，需要 groups 里的 indices 映射回全局
            # 这里 _worker 返回的 key (symbols_str) 用来查找 groups
            # 但 _worker 的返回值里第一项是 symbols_str
            pass 
            
            # 修正：imap_unordered 返回结果，我们需要在下面处理
            # 为了方便，我们在 worker 里不返回 indices 内容，只返回 index 列表
            # 这里逻辑有点绕，直接在 worker 返回 global indices 比较难，
            # 还是保持原来的逻辑：worker 返回局部 index，这里映射。
            
            # 这里的 _ 占位符实际上是 worker 返回的 symbols_str
            # 我们需要捕获它
        pass

    # 由于 imap_unordered 的写法，重写一下循环
    with Pool(nproc) as pool:
        for res in tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks), desc=desc_str):
            symbols_str, u_lbl, u_unlbl, t_lbl, t_unlbl = res
            
            indices = np.array(groups[symbols_str]["indices"])
            
            uniq_label_all.extend(indices[u_lbl].tolist())
            uniq_unlabel_all.extend(indices[u_unlbl].tolist())
            test_label_all.extend(indices[t_lbl].tolist())
            test_unlabel_all.extend(indices[t_unlbl].tolist())

    return uniq_label_all, uniq_unlabel_all, test_label_all, test_unlabel_all

def _worker(args):
    symbols_str, cache_dir, mode, param = args
    soap, idx_map, has_label = load_cache_with_label(symbols_str, cache_dir)
    
    if mode == 'fps':
        # param is n_select
        uniq_label_idx, uniq_unlabel_idx, test_label_idx, test_unlabel_idx = \
            fps_selection_target_n(soap, has_label, param)
    else:
        # param is simlT
        uniq_label_idx, uniq_unlabel_idx, test_label_idx, test_unlabel_idx = \
            cosine_threshold_dedup(soap, has_label, param)
            
    return symbols_str, uniq_label_idx, uniq_unlabel_idx, test_label_idx, test_unlabel_idx
