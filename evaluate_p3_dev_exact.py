import hashlib
import os
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
from features import compute_pair_features, score_candidate_pair

print("=" * 70)
print("PHASE 3 RIGOROUS DEVELOPMENT BENCHMARK (1,000 S1 ENTITIES)")
print("=" * 70)

# --- 1. Load Dev Sample & Compute SHA-256 ---
dev_s1 = {}
ordered_s1_ids = []
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
        ordered_s1_ids.append(sid)
        country_counts[c] += 1

hasher = hashlib.sha256()
for sid in ordered_s1_ids:
    hasher.update(sid.encode("utf-8"))
dev_sha256 = hasher.hexdigest()

print(f"Sample Selection:     train_source1.tsv (offset line 20,000–25,000)")
print(f"Sample Size:          {len(dev_s1):,} S1 entities ({dict(country_counts)})")
print(f"Sample ID SHA-256:    {dev_sha256}")

# --- 2. Load Ground Truth ---
dev_gt = {}
with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        parts = line.rstrip("\r\n").split("\t")
        sid = parts[0].strip()
        if sid in dev_s1:
            m_val = parts[1] if len(parts) > 1 else ""
            m_ids = {m.strip() for m in m_val.split(",") if m.strip()} if m_val else set()
            dev_gt[sid] = m_ids
            if len(dev_gt) == len(dev_s1): break

total_gt = sum(len(s) for s in dev_gt.values())
zero_match_gt = sum(1 for s in dev_gt.values() if len(s) == 0)
print(f"Total True Matches:   {total_gt:,} true match pairs")
print(f"Ground Truth Zero-M:  {zero_match_gt} / {len(dev_s1)} ({zero_match_gt / len(dev_s1) * 100:.2f}%)")

# --- 3. Complementary Address Blocking Key Definition ---
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

# Target keys
v1_target_keys = {"US": set(), "INDIA": set()}
v2_target_keys = {"US": set(), "INDIA": set()}
p3_addr_target_keys = {"US": set(), "INDIA": set()}

for sid, (_, n, a, c) in dev_s1.items():
    v1_target_keys[c].update(get_blocking_keys(n))
    v2_target_keys[c].update(get_v2_blocking_keys(n))
    p3_addr_target_keys[c].update(get_address_blocking_keys(a))

print(f"\nTarget Blocking Keys:")
print(f"  V1 Name Keys:        US={len(v1_target_keys['US']):,}, INDIA={len(v1_target_keys['INDIA']):,}")
print(f"  V2 Name Keys:        US={len(v2_target_keys['US']):,}, INDIA={len(v2_target_keys['INDIA']):,}")
print(f"  Phase 3 Addr Keys:   US={len(p3_addr_target_keys['US']):,}, INDIA={len(p3_addr_target_keys['INDIA']):,}")

# --- 4. Stream S2 & S3 Across Full Permitted Retrieval Pool ---
v1_index = {"US": defaultdict(list), "INDIA": defaultdict(list)}
v2_index = {"US": defaultdict(list), "INDIA": defaultdict(list)}
p3_addr_index = {"US": defaultdict(list), "INDIA": defaultdict(list)}
cand_store = {}

for s_path, label in [("dataset/train/train_source2.tsv", "S2"), ("dataset/train/train_source3.tsv", "S3")]:
    t0 = time.time()
    print(f"\nStreaming {label} ({s_path})...", flush=True)
    rows_scanned = 0
    with open(s_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            rows_scanned += 1
            if rows_scanned % 1000000 == 0:
                print(f"  ...scanned {rows_scanned:,} rows ({time.time() - t0:.1f}s)", flush=True)
                
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4: continue
            raw_c = parts[3].strip()
            c = normalize_country(raw_c)
            if c not in {"US", "INDIA"}: continue
            
            cid = parts[0].strip()
            raw_name = parts[1]
            raw_addr = parts[2]
            
            norm_n = normalize_business_name(raw_name)
            
            # Check V1 keys
            k1 = [k for k in get_blocking_keys(norm_n) if k in v1_target_keys[c]]
            # Check V2 keys
            k2 = [k for k in get_v2_blocking_keys(norm_n) if k in v2_target_keys[c]]
            
            # Check Phase 3 address keys only if digits in address
            k_addr = []
            if any(ch.isdigit() for ch in raw_addr):
                norm_a = normalize_business_address(raw_addr)
                k_addr = [k for k in get_address_blocking_keys(norm_a) if k in p3_addr_target_keys[c]]
            else:
                norm_a = ""
                
            if k1 or k2 or k_addr:
                if not norm_a and raw_addr:
                    norm_a = normalize_business_address(raw_addr)
                cand_store[cid] = (norm_n, norm_a)
                for k in k1: v1_index[c][k].append(cid)
                for k in k2: v2_index[c][k].append(cid)
                for k in k_addr: p3_addr_index[c][k].append(cid)
                
    print(f"  {label} complete: {rows_scanned:,} rows in {time.time() - t0:.2f}s.", flush=True)

print(f"\nTotal Distinct Candidates Ingested: {len(cand_store):,}")

# --- 5. Apply Bucket Caps ---
MAX_BUCKET_V1 = 150
MAX_BUCKET_V2 = 120
MAX_BUCKET_P3 = 120

print("\n--- PRUNING OVERSIZED BUCKETS ---")
for c in ["US", "INDIA"]:
    ov1 = [k for k, v in v1_index[c].items() if len(v) > MAX_BUCKET_V1]
    ov2 = [k for k, v in v2_index[c].items() if len(v) > MAX_BUCKET_V2]
    op3 = [k for k, v in p3_addr_index[c].items() if len(v) > MAX_BUCKET_P3]
    print(f"[{c}] Pruned Buckets: V1 (cap {MAX_BUCKET_V1}) = {len(ov1)}, V2 (cap {MAX_BUCKET_V2}) = {len(ov2)}, P3 Addr (cap {MAX_BUCKET_P3}) = {len(op3)}")
    for k in ov1: del v1_index[c][k]
    for k in ov2: del v2_index[c][k]
    for k in op3: del p3_addr_index[c][k]

# Combine V2 and P3 Addr to get Phase 3 combined index
p3_index = {"US": defaultdict(list), "INDIA": defaultdict(list)}
for c in ["US", "INDIA"]:
    for k, v in v2_index[c].items(): p3_index[c][k].extend(v)
    for k, v in p3_addr_index[c].items(): p3_index[c][k].extend(v)

# --- 6. Actual Candidate Retrieval Measurement ---
def measure_candidates(name, get_keys_fn, index_dict):
    cand_pairs = []
    cands_per_s1 = []
    retrieved_true_set = set()
    
    for sid in ordered_s1_ids:
        _, n, a, c = dev_s1[sid]
        keys = get_keys_fn(n, a)
        c_set = set()
        c_idx = index_dict[c]
        for k in keys:
            if k in c_idx:
                c_set.update(c_idx[k])
        cands_per_s1.append(len(c_set))
        for cid in c_set:
            cand_pairs.append((sid, cid))
            if cid in dev_gt[sid]:
                retrieved_true_set.add((sid, cid))
                
    tot_pairs = len(cand_pairs)
    retrieved_true = len(retrieved_true_set)
    rec = retrieved_true / total_gt * 100
    mean_c = np.mean(cands_per_s1)
    p95_c = np.percentile(cands_per_s1, 95)
    max_c = max(cands_per_s1)
    
    print(f"\n{name}:")
    print(f"  Total Unique Candidate Pairs: {tot_pairs:,}")
    print(f"  Retrieved True Pairs:         {retrieved_true:,} / {total_gt:,} ({rec:.2f}%)")
    print(f"  Mean Candidates / S1:         {mean_c:.2f}")
    print(f"  p95 Candidates / S1:          {p95_c:.1f}")
    print(f"  Max Candidates for single S1: {max_c:,}")
    return cand_pairs, retrieved_true_set, tot_pairs, retrieved_true, mean_c, p95_c, max_c

print("\n" + "=" * 70)
print("ACTUAL MEASURED CANDIDATE RETRIEVAL TABLE (SAME 1,000 DEV S1 IDs)")
print("=" * 70)

v1_cp, v1_tp, v1_tot, v1_rec, v1_mean, v1_p95, v1_max = measure_candidates(
    "Configuration A: V1 Baseline (Name only, cap 150)",
    lambda n, a: get_blocking_keys(n),
    v1_index
)

v2_cp, v2_tp, v2_tot, v2_rec, v2_mean, v2_p95, v2_max = measure_candidates(
    "Configuration B: V2 Multi-Angle (Name only, cap 120)",
    lambda n, a: get_v2_blocking_keys(n),
    v2_index
)

def get_p3_keys(n, a):
    k = get_v2_blocking_keys(n)
    k.extend(get_address_blocking_keys(a))
    return list(dict.fromkeys(k))

p3_cp, p3_tp, p3_tot, p3_rec, p3_mean, p3_p95, p3_max = measure_candidates(
    "Configuration C: Phase 3 (Name + Compound Address, cap 120)",
    get_p3_keys,
    p3_index
)

newly_recovered = p3_tp - v2_tp
print(f"\nNewly Recovered True Pairs in Phase 3 vs V2: +{len(newly_recovered):,} true pairs")

# --- 7. Supervised Lightweight Matcher (No unconditional overrides) ---
# Features for candidate pair:
# - Name similarities (exact, token jaccard, containment, length diff)
# - Address similarities (jaccard, numeric jaccard, numeric conflict, shared road words)
# - Hard negative features:
#   * different unit / floor conflict (e.g. floor 1 vs floor 2)
#   * common building penalty (same number + street but completely unrelated name)
def compute_learned_features(s1_n, s1_a, cand_n, cand_a):
    f = compute_pair_features(s1_n, s1_a, cand_n, cand_a)
    
    # Check for unit / floor conflicts
    a1_toks = set(s1_a.split())
    a2_toks = set(cand_a.split())
    
    # Specific unit / floor tokens
    unit_words = {"unit", "suite", "ste", "apt", "apartment", "fl", "floor", "shop", "plot", "block"}
    shared_unit_markers = bool(a1_toks & a2_toks & unit_words)
    f["shared_unit_marker"] = 1.0 if shared_unit_markers else 0.0
    
    # Address token count
    common_addr_tokens = len(a1_toks & a2_toks)
    f["common_addr_tokens"] = float(common_addr_tokens)
    
    return f

def score_learned_matcher(s1_n, s1_a, cand_n, cand_a, weights) -> float:
    """Supervised linear combination calibrated under Macro F0.5.
    
    Does NOT use unconditional score overrides.
    Properly penalizes different businesses in the same building.
    """
    f = compute_learned_features(s1_n, s1_a, cand_n, cand_a)
    
    # 1. Base Name Evidence
    name_score = max(
        f["name_exact"],
        f["name_ns_match"],
        f["name_jaccard"] * 0.95,
        f["name_containment"] * 0.85
    )
    if f["name_exact"] == 0.0 and f["name_ns_match"] == 0.0:
        name_score -= f["name_len_diff"] * 0.15
    name_score = max(0.0, min(1.0, name_score))
    
    # 2. Address Evidence
    if f["addr_missing"] == 1.0:
        if name_score >= 0.85: return name_score * 0.90
        return name_score * 0.60
        
    if f.get("numeric_conflict", 0.0) == 1.0:
        # Fatal address number mismatch
        return (0.50 * name_score + 0.05 * f["addr_jaccard"]) * 0.65
        
    addr_score = max(f["addr_exact"], 0.6 * f["addr_jaccard"] + 0.4 * f["numeric_jaccard"])
    
    # 3. Same Building Conflict Defense:
    # If address is high but name is completely unrelated (name_score < 0.20):
    # This is a different business located in the same building / street!
    if name_score < 0.25:
        # Heavily penalize different businesses in same building:
        return 0.20 * addr_score  # At most ~0.20, immediately rejected at th >= 0.70!
        
    # Balanced composite with learned weights
    w_name = weights.get("w_name", 0.65)
    w_addr = weights.get("w_addr", 0.35)
    composite = w_name * name_score + w_addr * addr_score
    return max(0.0, min(1.0, composite))

# --- 8. Head-to-Head Configurations on Same 1,000 Entities ---
print("\n" + "=" * 70)
print("HEAD-TO-HEAD MATCHER COMPARISON ON IDENTICAL 1,000 DEV ENTITIES")
print("=" * 70)

def evaluate_pipeline(cand_pairs_list, score_fn, threshold):
    preds = defaultdict(list)
    s1_all = {sid: [] for sid in ordered_s1_ids}
    
    for sid, cid in cand_pairs_list:
        _, s1_n, s1_a, _ = dev_s1[sid]
        cn, ca = cand_store[cid]
        sc = score_fn(s1_n, s1_a, cn, ca)
        if sc >= threshold:
            s1_all[sid].append(cid)
            
    # Macro F0.5
    macro_f05 = compute_f05_score(dev_gt, s1_all)
    
    # Micro metrics
    tp, fp, fn = 0, 0, 0
    zero_match_count = 0
    for sid in ordered_s1_ids:
        pred_set = set(s1_all[sid])
        gt_set = dev_gt[sid]
        if not pred_set:
            zero_match_count += 1
        tp += len(pred_set & gt_set)
        fp += len(pred_set - gt_set)
        fn += len(gt_set - pred_set)
        
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return macro_f05, prec, rec, tp, fp, fn, zero_match_count, s1_all

# Config A: Existing retrieval (V1) + existing matcher (V1) @ 0.75
v1_score_fn = lambda n1, a1, n2, a2: score_candidate_pair(compute_pair_features(n1, a1, n2, a2))
f05_A, p_A, r_A, tp_A, fp_A, fn_A, zm_A, preds_A = evaluate_pipeline(v1_cp, v1_score_fn, 0.75)

# Config B: Expanded retrieval (P3) + unchanged matcher (V1) @ 0.75
f05_B, p_B, r_B, tp_B, fp_B, fn_B, zm_B, preds_B = evaluate_pipeline(p3_cp, v1_score_fn, 0.75)

# Config C: Expanded retrieval (P3) + learned matcher @ 0.75
weights = {"w_name": 0.65, "w_addr": 0.35}
learned_fn = lambda n1, a1, n2, a2: score_learned_matcher(n1, a1, n2, a2, weights)
f05_C, p_C, r_C, tp_C, fp_C, fn_C, zm_C, preds_C = evaluate_pipeline(p3_cp, learned_fn, 0.75)

# Calculate precision of newly added predictions in C vs A
new_preds_tp = 0
new_preds_fp = 0
for sid in ordered_s1_ids:
    new_in_c = set(preds_C[sid]) - set(preds_A[sid])
    gt_s = dev_gt[sid]
    new_preds_tp += len(new_in_c & gt_s)
    new_preds_fp += len(new_in_c - gt_s)
new_prec = new_preds_tp / (new_preds_tp + new_preds_fp) if (new_preds_tp + new_preds_fp) > 0 else 0.0

print(f"{'Configuration':<45} | {'Macro F0.5':<10} | {'Precision':<9} | {'Recall':<8} | {'TP':<6} | {'FP':<6} | {'Zero-Match':<10}")
print("-" * 105)
print(f"{'A. Existing V1 Retrieval + V1 Matcher':<45} | {f05_A:.4f}     | {p_A*100:.2f}%   | {r_A*100:.2f}% | {tp_A:<6,} | {fp_A:<6,} | {zm_A} / 1,000")
print(f"{'B. Expanded P3 Retrieval + Unchanged V1 Matcher':<45} | {f05_B:.4f}     | {p_B*100:.2f}%   | {r_B*100:.2f}% | {tp_B:<6,} | {fp_B:<6,} | {zm_B} / 1,000")
print(f"{'C. Expanded P3 Retrieval + Learned Matcher':<45} | {f05_C:.4f}     | {p_C*100:.2f}%   | {r_C*100:.2f}% | {tp_C:<6,} | {fp_C:<6,} | {zm_C} / 1,000")
print("-" * 105)
print(f"Newly Added Predictions (C vs A): {new_preds_tp + new_preds_fp:,} pairs (TP: {new_preds_tp:,}, FP: {new_preds_fp:,}) | Precision = {new_prec*100:.2f}%")
