import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
src_dir = Path("code/business_entity_resolution/src").resolve()
sys.path.insert(0, str(src_dir))

from normalization import normalize_business_name, normalize_business_address, normalize_country
from blocking import get_blocking_keys
from predict_v2 import get_v2_blocking_keys
from evaluation import compute_f05_score

print("=" * 65)
print("PHASE 3 FAST CANDIDATE RETRIEVAL & MATCHER BENCHMARK")
print("1,000 Dev Entities (train_source1.tsv, lines 20,000 to 25,000)")
print("=" * 65)

# --- 1. Address Blocking Key Generator ---
def get_address_blocking_keys(norm_addr: str) -> list:
    if not norm_addr: return []
    tokens = norm_addr.split()
    nums = [t for t in tokens if any(c.isdigit() for c in t)]
    words = [t for t in tokens if not any(c.isdigit() for c in t) and len(t) >= 3 and t not in {
        "street", "road", "avenue", "lane", "drive", "court", "circle", "boulevard",
        "near", "opp", "opposite", "plot", "number", "door", "floor", "building", "phase"
    }]
    keys = []
    if nums and words:
        primary_num = nums[0]
        keys.append(f"addr:{primary_num}_{words[0]}")
        if len(words) >= 2:
            keys.append(f"addr:{primary_num}_{words[1]}")
    return keys

def get_phase3_blocking_keys(norm_name: str, norm_addr: str) -> list:
    keys = get_v2_blocking_keys(norm_name)
    keys.extend(get_address_blocking_keys(norm_addr))
    return list(dict.fromkeys(keys))

# --- 2. Load 1,000 Dev S1 Entities ---
dev_s1 = {}
country_counts = Counter()
with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
    f.readline()
    for idx, line in enumerate(f):
        if idx < 20000: continue
        if len(dev_s1) >= 1000: break
        parts = line.rstrip("\r\n").split("\t")
        if len(parts) < 4: continue
        sid = parts[0].strip()
        c = normalize_country(parts[3])
        if c not in {"US", "INDIA"}: continue
        n = normalize_business_name(parts[1])
        a = normalize_business_address(parts[2])
        dev_s1[sid] = (sid, n, a, c)
        country_counts[c] += 1

dev_ids = set(dev_s1.keys())

# --- 3. Load Ground Truth ---
dev_gt = {}
with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        parts = line.rstrip("\r\n").split("\t")
        sid = parts[0].strip()
        if sid in dev_ids:
            m_val = parts[1] if len(parts) > 1 else ""
            m_ids = {m.strip() for m in m_val.split(",") if m.strip()} if m_val else set()
            dev_gt[sid] = m_ids
            if len(dev_gt) == len(dev_ids): break

total_gt = sum(len(s) for s in dev_gt.values())
print(f"Loaded {len(dev_s1):,} Dev S1 entities ({dict(country_counts)}) with {total_gt:,} true match pairs.")

v1_target_keys = {"US": set(), "INDIA": set()}
v2_target_keys = {"US": set(), "INDIA": set()}
p3_target_keys = {"US": set(), "INDIA": set()}

for sid, (_, n, a, c) in dev_s1.items():
    v1_target_keys[c].update(get_blocking_keys(n))
    v2_target_keys[c].update(get_v2_blocking_keys(n))
    p3_target_keys[c].update(get_phase3_blocking_keys(n, a))

print(f"Target Keys:")
print(f"  V1:     US={len(v1_target_keys['US']):,}, INDIA={len(v1_target_keys['INDIA']):,}")
print(f"  V2:     US={len(v2_target_keys['US']):,}, INDIA={len(v2_target_keys['INDIA']):,}")
print(f"  P3:     US={len(p3_target_keys['US']):,}, INDIA={len(p3_target_keys['INDIA']):,}")

# Build precise trigger lookups
target_name_words = {"US": set(), "INDIA": set()}
target_addr_nums = {"US": defaultdict(set), "INDIA": defaultdict(set)}

for c in ["US", "INDIA"]:
    for k in v2_target_keys[c]:
        if ":" in k:
            w = k.split(":", 1)[1]
            if len(w) >= 3:
                target_name_words[c].add(w)
                for tok in w.split():
                    if len(tok) >= 3:
                        target_name_words[c].add(tok)
    for k in p3_target_keys[c]:
        if k.startswith("addr:"):
            parts = k[5:].split("_", 1)
            if len(parts) == 2:
                num, word = parts
                target_addr_nums[c][num].add(word)

print(f"Trigger Lookups Prepared:")
for c in ["US", "INDIA"]:
    print(f"  [{c}] Name Triggers: {len(target_name_words[c]):,}, Address Num Triggers: {len(target_addr_nums[c]):,}")

# --- 5. High-Speed Candidate Streaming with Inverted Indexing ---
v1_index = {"US": defaultdict(list), "INDIA": defaultdict(list)}
v2_index = {"US": defaultdict(list), "INDIA": defaultdict(list)}
p3_index = {"US": defaultdict(list), "INDIA": defaultdict(list)}
p3_nocap_index = {"US": defaultdict(list), "INDIA": defaultdict(list)}
cand_store = {}

for s_path, label in [("dataset/train/train_source2.tsv", "S2"), ("dataset/train/train_source3.tsv", "S3")]:
    t0 = time.time()
    print(f"\nStreaming {label} ({s_path})...", flush=True)
    hits = 0
    total_lines = 0
    
    with open(s_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            total_lines += 1
            # Fast country check
            if "\tUS\n" in line or line.endswith("\tUS\r\n") or line.endswith("\tUS"):
                c = "US"
            elif "\tINDIA\n" in line or line.endswith("\tINDIA\r\n") or line.endswith("\tINDIA"):
                c = "INDIA"
            else:
                continue
                
            line_lower = line.lower()
            line_tokens = set(re.findall(r'[a-z0-9]+', line_lower))
            
            # Ultra-fast exact trigger check:
            # 1. Name word match?
            has_name = bool(line_tokens & target_name_words[c])
            # 2. Address (num + street word) match?
            matched_nums = line_tokens & target_addr_nums[c].keys()
            has_addr = any(line_tokens & target_addr_nums[c][num] for num in matched_nums) if matched_nums else False
            
            if not (has_name or has_addr):
                continue
                
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4: continue
            cid = parts[0].strip()
            cname = parts[1]
            caddr = parts[2]
            
            norm_n = normalize_business_name(cname)
            norm_a = normalize_business_address(caddr)
            
            k1 = [k for k in get_blocking_keys(norm_n) if k in v1_target_keys[c]]
            k2 = [k for k in get_v2_blocking_keys(norm_n) if k in v2_target_keys[c]]
            kp3 = [k for k in get_phase3_blocking_keys(norm_n, norm_a) if k in p3_target_keys[c]]
            
            if k1 or k2 or kp3:
                hits += 1
                cand_store[cid] = (norm_n, norm_a)
                for k in k1: v1_index[c][k].append(cid)
                for k in k2: v2_index[c][k].append(cid)
                for k in kp3: 
                    p3_index[c][k].append(cid)
                    p3_nocap_index[c][k].append(cid)
                    
    print(f"  {label}: scanned {total_lines:,} rows, matched {hits:,} candidates in {time.time() - t0:.2f}s.", flush=True)

print(f"\nTotal Distinct Candidates Indexed: {len(cand_store):,}")

# --- 6. Oversized Bucket Filtering Analysis ---
MAX_BUCKET_V1 = 150
MAX_BUCKET_V2 = 120
MAX_BUCKET_P3 = 120

print("\n--- OVERSIZED BUCKET DIAGNOSIS ---")
for c in ["US", "INDIA"]:
    ov1 = [k for k, v in v1_index[c].items() if len(v) > MAX_BUCKET_V1]
    ov2 = [k for k, v in v2_index[c].items() if len(v) > MAX_BUCKET_V2]
    op3 = [k for k, v in p3_index[c].items() if len(v) > MAX_BUCKET_P3]
    
    print(f"[{c}] Oversized buckets pruned:")
    print(f"  V1 (cap {MAX_BUCKET_V1}): {len(ov1)} buckets (top: {[f'{k}:{len(v1_index[c][k])}' for k in ov1[:3]]})")
    print(f"  V2 (cap {MAX_BUCKET_V2}): {len(ov2)} buckets (top: {[f'{k}:{len(v2_index[c][k])}' for k in ov2[:3]]})")
    print(f"  P3 (cap {MAX_BUCKET_P3}): {len(op3)} buckets (top: {[f'{k}:{len(p3_index[c][k])}' for k in op3[:3]]})")
    
    for k in ov1: del v1_index[c][k]
    for k in ov2: del v2_index[c][k]
    for k in op3: del p3_index[c][k]

# --- 7. Candidate Retrieval Evaluation ---
def eval_candidates(strategy_name, index_dict, key_fn):
    retrieved_true = 0
    cands_per_s1 = []
    pairs = set()
    
    for sid, (_, n, a, c) in dev_s1.items():
        keys = key_fn(n, a)
        c_ids = set()
        c_idx = index_dict[c]
        for k in keys:
            if k in c_idx:
                c_ids.update(c_idx[k])
        cands_per_s1.append(len(c_ids))
        true_m = c_ids & dev_gt[sid]
        retrieved_true += len(true_m)
        for cid in true_m:
            pairs.add((sid, cid))
            
    recall = retrieved_true / total_gt * 100
    total_c = sum(cands_per_s1)
    avg_c = np.mean(cands_per_s1)
    p95_c = np.percentile(cands_per_s1, 95)
    p99_c = np.percentile(cands_per_s1, 99)
    max_c = max(cands_per_s1)
    
    print(f"\n{strategy_name}:")
    print(f"  Candidate Recall:       {retrieved_true:,} / {total_gt:,} ({recall:.2f}%)")
    print(f"  Total Candidates:       {total_c:,}")
    print(f"  Avg Cands / S1:         {avg_c:.2f}")
    print(f"  p95 Cands / S1:         {p95_c:.1f}")
    print(f"  p99 Cands / S1:         {p99_c:.1f}")
    print(f"  Max Cands for single S1:{max_c:,}")
    return pairs, retrieved_true, total_c, avg_c, p95_c

v1_pairs, v1_rec, v1_tot, v1_avg, v1_p95 = eval_candidates("V1 Baseline (Name blocking, cap 150)", v1_index, lambda n, a: get_blocking_keys(n))
v2_pairs, v2_rec, v2_tot, v2_avg, v2_p95 = eval_candidates("V2 Multi-Angle (Name only, cap 120)", v2_index, lambda n, a: get_v2_blocking_keys(n))
p3_pairs, p3_rec, p3_tot, p3_avg, p3_p95 = eval_candidates("Phase 3 (Name + Address-Aware, cap 120)", p3_index, get_phase3_blocking_keys)
p3_nocap_pairs, p3_nocap_rec, _, _, _ = eval_candidates("Phase 3 NO CAP (No bucket cap pruning)", p3_nocap_index, get_phase3_blocking_keys)

newly_recovered = p3_pairs - v1_pairs
print("\n" + "=" * 65)
print("RETRIEVAL SUMMARY COMPARISON TABLE")
print("=" * 65)
print(f"{'Strategy':<35} | {'Cand Recall':<15} | {'Total Cands':<12} | {'Avg/S1':<8} | {'p95/S1':<8}")
print("-" * 88)
print(f"{'V1 Baseline (cap 150)':<35} | {v1_rec:,} ({v1_rec/total_gt*100:.2f}%)   | {v1_tot:<12,} | {v1_avg:<8.1f} | {v1_p95:<8.1f}")
print(f"{'V2 Multi-Angle (cap 120)':<35} | {v2_rec:,} ({v2_rec/total_gt*100:.2f}%)   | {v2_tot:<12,} | {v2_avg:<8.1f} | {v2_p95:<8.1f}")
print(f"{'Phase 3 (Name+Address, cap 120)':<35} | {p3_rec:,} ({p3_rec/total_gt*100:.2f}%)   | {p3_tot:<12,} | {p3_avg:<8.1f} | {p3_p95:<8.1f}")
print(f"{'Phase 3 (NO CAP)':<35} | {p3_nocap_rec:,} ({p3_nocap_rec/total_gt*100:.2f}%)   | {'N/A':<12} | {'N/A':<8} | {'N/A':<8}")
print("-" * 88)
print(f"Phase 3 vs V1 Recall Delta: +{p3_rec - v1_rec:,} true pairs (+{(p3_rec - v1_rec)/total_gt*100:.2f}% abs recall!)")
print(f"Newly Recovered Pairs vs V1: +{len(newly_recovered):,} true pairs")
print(f"Lost pairs due to bucket cap 120: {p3_nocap_rec - p3_rec:,} pairs ({(p3_nocap_rec - p3_rec)/total_gt*100:.2f}%)")

# --- 8. Train & Evaluate Shared Matcher ---
print("\n" + "=" * 65)
print("SHARED MATCHER EVALUATION ON DEVELOPMENT SPLIT")
print("=" * 65)

from predict_v3 import score_pair_v3

# Calibrate Phase 3 Matcher with Address-Dominant Matching Rule:
# If street number & address tokens are strongly identical (>=4 shared tokens),
# resolve transliterated or acronymized names that would otherwise fail name_score >= 0.40!
def score_pair_p3(s1_n: str, s1_a: str, cand_n: str, cand_a: str) -> float:
    # 1. First compute standard V3 score
    base_score = score_pair_v3(s1_n, s1_a, cand_n, cand_a)
    if base_score >= 0.70:
        return base_score
        
    # 2. Check for Address-Dominant Match (Transliterated / Multilingual / Acronym cases)
    if not s1_a or not cand_a:
        return base_score
        
    a1_toks = set(s1_a.split())
    a2_toks = set(cand_a.split())
    nums1 = {t for t in a1_toks if any(c.isdigit() for c in t)}
    nums2 = {t for t in a2_toks if any(c.isdigit() for c in t)}
    
    # Must share numeric tokens (door / plot / house / street number)
    if nums1 and nums2 and (nums1 & nums2):
        common_addr = len(a1_toks & a2_toks)
        # Require strong address overlap (shared number + at least 3 street/city tokens)
        if common_addr >= 4:
            # Address is verified identical. Now inspect name:
            # If name is acronym, prefix, or transliterated (non-empty)
            return max(base_score, 0.76)
            
    return base_score

# Evaluate across multiple decision thresholds
for th in [0.70, 0.72, 0.74, 0.75]:
    # Phase 3 Candidates with Phase 3 Matcher
    p3_preds = {}
    for sid, (_, n, a, c) in dev_s1.items():
        keys = get_phase3_blocking_keys(n, a)
        c_ids = set()
        for k in keys:
            if k in p3_index[c]:
                c_ids.update(p3_index[c][k])
        matches = []
        for cid in c_ids:
            cn, ca = cand_store[cid]
            if score_pair_p3(n, a, cn, ca) >= th:
                matches.append(cid)
        p3_preds[sid] = matches
    p3_f05 = compute_f05_score(dev_gt, p3_preds)
    
    # V1 Candidates with V1 Baseline Matcher
    from features import compute_pair_features, score_candidate_pair
    v1_preds = {}
    for sid, (_, n, a, c) in dev_s1.items():
        keys = get_blocking_keys(n)
        c_ids = set()
        for k in keys:
            if k in v1_index[c]:
                c_ids.update(v1_index[c][k])
        matches = []
        for cid in c_ids:
            cn, ca = cand_store[cid]
            f_dict = compute_pair_features(n, a, cn, ca)
            if score_candidate_pair(f_dict) >= th:
                matches.append(cid)
        v1_preds[sid] = matches
    v1_f05 = compute_f05_score(dev_gt, v1_preds)
    
    print(f"Threshold {th:.2f}:  V1 Matcher F0.5 = {v1_f05:.4f}  |  Phase 3 Matcher F0.5 = {p3_f05:.4f}  (Delta: {p3_f05 - v1_f05:+.4f})")
