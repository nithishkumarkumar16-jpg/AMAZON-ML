import hashlib
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
src_dir = Path("code/business_entity_resolution/src").resolve()
sys.path.insert(0, str(src_dir))

from normalization import normalize_business_name, normalize_business_address, normalize_country
from blocking import get_blocking_keys
from evaluation import compute_f05_score
from features import compute_pair_features, score_candidate_pair

# --- 1. Address Blocking Key Generator ---
def get_address_blocking_keys(norm_addr: str) -> list:
    if not norm_addr: return []
    tokens = norm_addr.split()
    nums = [t for t in tokens if any(c.isdigit() for c in t)]
    words = [t for t in tokens if not any(c.isdigit() for c in t) and len(t) >= 3 and t not in {
        "street", "road", "avenue", "lane", "drive", "court", "circle", "boulevard",
        "near", "opp", "opposite", "plot", "number", "door", "floor", "building", "phase",
        "first", "second", "third", "north", "south", "east", "west"
    }]
    keys = []
    if nums and words:
        primary_num = nums[0]
        keys.append(f"addr:{primary_num}_{words[0]}")
        if len(words) >= 2:
            keys.append(f"addr:{primary_num}_{words[1]}")
    return keys

def get_p3_keys(n: str, a: str) -> list:
    k = get_blocking_keys(n)
    k.extend(get_address_blocking_keys(a))
    return list(dict.fromkeys(k))

def score_v1(s1_n, s1_a, c_n, c_a):
    return score_candidate_pair(compute_pair_features(s1_n, s1_a, c_n, c_a))

def score_p3_rule_based(s1_n, s1_a, c_n, c_a):
    f = compute_pair_features(s1_n, s1_a, c_n, c_a)
    name_score = max(
        f["name_exact"],
        f["name_ns_match"],
        f["name_jaccard"] * 0.95,
        f["name_containment"] * 0.85
    )
    if f["name_exact"] == 0.0 and f["name_ns_match"] == 0.0:
        name_score -= f["name_len_diff"] * 0.15
    name_score = max(0.0, min(1.0, name_score))
    
    if f["addr_missing"] == 1.0:
        if name_score >= 0.85: return name_score * 0.90
        return name_score * 0.60
        
    if f.get("numeric_conflict", 0.0) == 1.0:
        return (0.50 * name_score + 0.05 * f["addr_jaccard"]) * 0.65
        
    addr_score = max(f["addr_exact"], 0.6 * f["addr_jaccard"] + 0.4 * f["numeric_jaccard"])
    
    # Building Conflict Defense
    if name_score < 0.25:
        return 0.20 * addr_score
        
    composite = 0.65 * name_score + 0.35 * addr_score
    return max(0.0, min(1.0, composite))


# --- 2. Parallel Chunk Processing Function ---
DIGIT_FINDER = re.compile(r"\d+")

def scan_chunk(args):
    filepath, start_byte, end_byte, conf_name_keys, conf_addr_keys, conf_addr_nums = args
    v1_postings = {"US": defaultdict(list), "INDIA": defaultdict(list)}
    p3_addr_postings = {"US": defaultdict(list), "INDIA": defaultdict(list)}
    cand_records = {}
    
    with open(filepath, "rb") as f:
        if start_byte != 0:
            f.seek(start_byte)
            f.readline()  # Skip partial line
        else:
            f.seek(0)
            f.readline()  # Skip TSV header
            
        while f.tell() < end_byte:
            raw_line = f.readline()
            if not raw_line:
                break
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                line = raw_line.decode("utf-8", errors="replace")
                
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4: continue
            raw_c = parts[3].strip().upper()
            if raw_c not in {"US", "INDIA"}: continue
            
            raw_name = parts[1]
            raw_addr = parts[2]
            
            norm_n = normalize_business_name(raw_name)
            k1 = [k for k in get_blocking_keys(norm_n) if k in conf_name_keys[raw_c]]
            
            k_addr = []
            nums_in_addr = DIGIT_FINDER.findall(raw_addr)
            if any(num in conf_addr_nums[raw_c] for num in nums_in_addr):
                norm_a = normalize_business_address(raw_addr)
                k_addr = [k for k in get_address_blocking_keys(norm_a) if k in conf_addr_keys[raw_c]]
            else:
                norm_a = ""
                
            if k1 or k_addr:
                cid = parts[0].strip()
                if not norm_a and raw_addr:
                    norm_a = normalize_business_address(raw_addr)
                cand_records[cid] = (norm_n, norm_a)
                for k in k1: v1_postings[raw_c][k].append(cid)
                for k in k_addr: p3_addr_postings[raw_c][k].append(cid)
                
    return v1_postings, p3_addr_postings, cand_records


def main():
    print("=" * 75, flush=True)
    print("PARALLEL CONFIRMATION BENCHMARK ON UNTOUCHED 1,000 S1 ENTITIES", flush=True)
    print("=" * 75, flush=True)

    # 1. Extract Confirmation Sample (Rows 120,000 to 135,000)
    conf_s1 = {}
    conf_ordered_ids = []
    conf_country_counts = Counter()

    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for idx, line in enumerate(f):
            if idx < 120000: continue
            if len(conf_s1) >= 1000: break
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4: continue
            sid = parts[0].strip()
            c = normalize_country(parts[3])
            if c not in {"US", "INDIA"}: continue
            n = normalize_business_name(parts[1])
            a = normalize_business_address(parts[2])
            conf_s1[sid] = (sid, n, a, c)
            conf_ordered_ids.append(sid)
            conf_country_counts[c] += 1

    conf_sha = hashlib.sha256("".join(conf_ordered_ids).encode("utf-8")).hexdigest()
    print(f"Confirmation Sample Size:     {len(conf_s1):,} entities ({dict(conf_country_counts)})", flush=True)
    print(f"Confirmation Sample SHA-256:  {conf_sha}", flush=True)

    # 2. Load Ground Truth
    conf_gt = {}
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            sid = parts[0].strip()
            if sid in conf_s1:
                m_val = parts[1] if len(parts) > 1 else ""
                m_ids = {m.strip() for m in m_val.split(",") if m.strip()} if m_val else set()
                conf_gt[sid] = m_ids
                if len(conf_gt) == len(conf_s1): break

    total_gt = sum(len(s) for s in conf_gt.values())
    zero_m_gt = sum(1 for s in conf_gt.values() if len(s) == 0)
    print(f"Ground Truth Total Matches:   {total_gt:,} true match pairs", flush=True)
    print(f"Ground Truth Zero-Matches:    {zero_m_gt} / {len(conf_s1)} ({zero_m_gt/len(conf_s1)*100:.2f}%)", flush=True)

    # 3. Target Keys for Conf Sample
    conf_name_keys = {"US": set(), "INDIA": set()}
    conf_addr_keys = {"US": set(), "INDIA": set()}
    conf_addr_nums = {"US": set(), "INDIA": set()}

    for sid, (_, n, a, c) in conf_s1.items():
        conf_name_keys[c].update(get_blocking_keys(n))
        ak = get_address_blocking_keys(a)
        conf_addr_keys[c].update(ak)
        for k in ak:
            num = k.split(":")[1].split("_")[0]
            conf_addr_nums[c].add(num)

    print(f"Target Keys: Name={len(conf_name_keys['US']) + len(conf_name_keys['INDIA']):,}, Addr={len(conf_addr_keys['US']) + len(conf_addr_keys['INDIA']):,}", flush=True)

    # 4. Multi-Worker Parallel Scan across S2 and S3
    NUM_WORKERS = 8
    tasks = []
    
    for fpath in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
        fsize = os.path.getsize(fpath)
        chunk_size = fsize // NUM_WORKERS
        for i in range(NUM_WORKERS):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < NUM_WORKERS - 1 else fsize
            tasks.append((fpath, start, end, conf_name_keys, conf_addr_keys, conf_addr_nums))

    print(f"\nLaunching {len(tasks)} parallel scan tasks across {NUM_WORKERS} workers...", flush=True)
    t0_scan = time.time()
    
    v1_index = {"US": defaultdict(list), "INDIA": defaultdict(list)}
    p3_addr_index = {"US": defaultdict(list), "INDIA": defaultdict(list)}
    cand_store = {}

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for idx, (v1_p, p3_p, c_rec) in enumerate(executor.map(scan_chunk, tasks)):
            cand_store.update(c_rec)
            for c in ["US", "INDIA"]:
                for k, v in v1_p[c].items(): v1_index[c][k].extend(v)
                for k, v in p3_p[c].items(): p3_addr_index[c][k].extend(v)
            print(f"  Completed chunk {idx+1}/{len(tasks)} ({time.time() - t0_scan:.1f}s)", flush=True)

    print(f"Parallel scan complete in {time.time() - t0_scan:.2f}s! Distinct candidates: {len(cand_store):,}", flush=True)

    # 5. Apply Bucket Caps
    MAX_BUCKET_V1 = 150
    MAX_BUCKET_P3 = 120

    v1_pruned = {"US": {k: v for k, v in v1_index["US"].items() if len(v) <= MAX_BUCKET_V1},
                 "INDIA": {k: v for k, v in v1_index["INDIA"].items() if len(v) <= MAX_BUCKET_V1}}

    p3_addr_pruned = {"US": {k: v for k, v in p3_addr_index["US"].items() if len(v) <= MAX_BUCKET_P3},
                      "INDIA": {k: v for k, v in p3_addr_index["INDIA"].items() if len(v) <= MAX_BUCKET_P3}}

    p3_combined = {"US": defaultdict(list), "INDIA": defaultdict(list)}
    for c in ["US", "INDIA"]:
        for k, v in v1_pruned[c].items():
            if len(v) <= MAX_BUCKET_P3:
                p3_combined[c][k].extend(v)
        for k, v in p3_addr_pruned[c].items():
            p3_combined[c][k].extend(v)

    # 6. Evaluate Pipelines
    def run_eval(key_fn, index_dict, score_fn, threshold):
        cand_pairs_count = 0
        retrieved_true = 0
        preds_dict = defaultdict(list)
        s1_all = {sid: [] for sid in conf_ordered_ids}
        
        for sid in conf_ordered_ids:
            _, s1_n, s1_a, c = conf_s1[sid]
            keys = key_fn(s1_n, s1_a)
            c_set = set()
            c_idx = index_dict[c]
            for k in keys:
                if k in c_idx:
                    c_set.update(c_idx[k])
            cand_pairs_count += len(c_set)
            retrieved_true += len(c_set & conf_gt[sid])
            
            for cid in c_set:
                cn, ca = cand_store[cid]
                sc = score_fn(s1_n, s1_a, cn, ca)
                if sc >= threshold:
                    s1_all[sid].append(cid)
                    preds_dict[sid].append(cid)
                    
        f05 = compute_f05_score(conf_gt, {sid: set(v) for sid, v in s1_all.items()})
        
        tp, fp, fn = 0, 0, 0
        zm = 0
        for sid in conf_ordered_ids:
            pset = set(s1_all[sid])
            gset = conf_gt[sid]
            if not pset: zm += 1
            tp += len(pset & gset)
            fp += len(pset - gset)
            fn += len(gset - pset)
            
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        c_rec = retrieved_true / total_gt * 100
        
        return f05, p, r, c_rec, tp, fp, fn, zm, cand_pairs_count, retrieved_true, preds_dict

    # V1 Baseline
    f05_v1, p_v1, r_v1, cr_v1, tp_v1, fp_v1, fn_v1, zm_v1, cp_v1, rt_v1, preds_v1 = run_eval(
        lambda n, a: get_blocking_keys(n),
        v1_pruned,
        score_v1,
        0.75
    )

    # Phase 3 Rule-Based
    f05_p3, p_p3, r_p3, cr_p3, tp_p3, fp_p3, fn_p3, zm_p3, cp_p3, rt_p3, preds_p3 = run_eval(
        get_p3_keys,
        p3_combined,
        score_p3_rule_based,
        0.75
    )

    print("\n" + "=" * 75, flush=True)
    print("FINAL CONFIRMATION BENCHMARK ON UNTOUCHED 1,000 S1 ENTITIES", flush=True)
    print("=" * 75, flush=True)
    print(f"{'Configuration':<38} | {'Macro F0.5':<10} | {'Precision':<9} | {'Recall':<8} | {'Cand Rec':<9} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Zero-Match':<10}", flush=True)
    print("-" * 118, flush=True)
    print(f"{'V1 Baseline (Name only, Matcher @ 0.75)':<38} | {f05_v1:.4f}     | {p_v1*100:.2f}%   | {r_v1*100:.2f}% | {cr_v1:.2f}%    | {tp_v1:<6} | {fp_v1:<6} | {fn_v1:<6} | {zm_v1} / {len(conf_s1)}", flush=True)
    print(f"{'Phase 3 (Name+Addr, Rule-Based @ 0.75)':<38} | {f05_p3:.4f}     | {p_p3*100:.2f}%   | {r_p3*100:.2f}% | {cr_p3:.2f}%    | {tp_p3:<6} | {fp_p3:<6} | {fn_p3:<6} | {zm_p3} / {len(conf_s1)}", flush=True)
    print("-" * 118, flush=True)
    print(f"Delta Macro F0.5:   {f05_p3 - f05_v1:+.4f}", flush=True)
    print(f"Delta Precision:    {(p_p3 - p_v1)*100:+.2f}%", flush=True)
    print(f"Delta Recall:       {(r_p3 - r_v1)*100:+.2f}%", flush=True)
    print(f"Delta Cand Recall:  {(cr_p3 - cr_v1):+.2f}% (+{rt_p3 - rt_v1} true pairs retrieved)", flush=True)

if __name__ == "__main__":
    main()
