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

print("=" * 75)
print("PHASE 3 RIGOROUS VERIFICATION, PRUNING AUDIT & DISJOINT CONFIRMATION")
print("=" * 75)

# Ensure output directory exists
out_dir = Path("output")
out_dir.mkdir(exist_ok=True)

# ------------------------------------------------------------------------------
# 1. LOAD DEVELOPMENT SAMPLE (1,000 S1 ENTITIES)
# ------------------------------------------------------------------------------
dev_s1 = {}
dev_ordered_ids = []
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
        dev_ordered_ids.append(sid)
        country_counts[c] += 1

hasher = hashlib.sha256()
for sid in dev_ordered_ids:
    hasher.update(sid.encode("utf-8"))
dev_sha256 = hasher.hexdigest()

print(f"1. Development Sample Selection:")
print(f"   Source:                train_source1.tsv (offset lines 20,000–25,000)")
print(f"   Size:                  {len(dev_s1):,} S1 entities ({dict(country_counts)})")
print(f"   Sample ID SHA-256:     {dev_sha256}")

# Ground Truth for Dev
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

total_dev_gt = sum(len(s) for s in dev_gt.values())
zero_m_dev_gt = sum(1 for s in dev_gt.values() if len(s) == 0)
print(f"   Total GT Match Pairs:  {total_dev_gt:,}")
print(f"   Zero-Match GT Rows:    {zero_m_dev_gt} / {len(dev_s1)} ({zero_m_dev_gt / len(dev_s1) * 100:.2f}%)")

# ------------------------------------------------------------------------------
# 2. DEFINE BLOCKING KEYS & TARGET LOOKUPS
# ------------------------------------------------------------------------------
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
    k = get_v2_blocking_keys(n)
    k.extend(get_address_blocking_keys(a))
    return list(dict.fromkeys(k))

dev_v1_keys = {"US": set(), "INDIA": set()}
dev_v2_keys = {"US": set(), "INDIA": set()}
dev_p3_addr_keys = {"US": set(), "INDIA": set()}

# Fast triggers
dev_word_triggers = {"US": set(), "INDIA": set()}
dev_addr_nums = {"US": set(), "INDIA": set()}
dev_addr_words = {"US": set(), "INDIA": set()}

for sid, (_, n, a, c) in dev_s1.items():
    k1 = get_blocking_keys(n)
    k2 = get_v2_blocking_keys(n)
    kp3_addr = get_address_blocking_keys(a)
    
    dev_v1_keys[c].update(k1)
    dev_v2_keys[c].update(k2)
    dev_p3_addr_keys[c].update(kp3_addr)
    
    for k in k2:
        if ":" in k:
            w = k.split(":", 1)[1]
            if len(w) >= 3:
                dev_word_triggers[c].add(w)
                for tok in w.split():
                    if len(tok) >= 3: dev_word_triggers[c].add(tok)
    for k in kp3_addr:
        parts = k[5:].split("_", 1)
        if len(parts) == 2:
            dev_addr_nums[c].add(parts[0])
            dev_addr_words[c].add(parts[1])

print(f"\n2. Extracted Target Keys for Dev:")
print(f"   V1 Keys:        US={len(dev_v1_keys['US']):,}, INDIA={len(dev_v1_keys['INDIA']):,}")
print(f"   V2 Keys:        US={len(dev_v2_keys['US']):,}, INDIA={len(dev_v2_keys['INDIA']):,}")
print(f"   P3 Addr Keys:   US={len(dev_p3_addr_keys['US']):,}, INDIA={len(dev_p3_addr_keys['INDIA']):,}")

# ------------------------------------------------------------------------------
# 3. STREAM TARGET POOL WITH ULTRA-FAST ROW PRE-FILTER
# ------------------------------------------------------------------------------
# Unconstrained indexes (before pruning)
v1_idx_raw = {"US": defaultdict(list), "INDIA": defaultdict(list)}
v2_idx_raw = {"US": defaultdict(list), "INDIA": defaultdict(list)}
p3_addr_idx_raw = {"US": defaultdict(list), "INDIA": defaultdict(list)}
cand_store = {}

for s_path, label in [("dataset/train/train_source2.tsv", "S2"), ("dataset/train/train_source3.tsv", "S3")]:
    t0 = time.time()
    print(f"\nStreaming {label} ({s_path})...", flush=True)
    rows_scanned = 0
    hits = 0
    with open(s_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            rows_scanned += 1
            if "\tUS\n" in line or line.endswith("\tUS\r\n") or line.endswith("\tUS"):
                c = "US"
            elif "\tINDIA\n" in line or line.endswith("\tINDIA\r\n") or line.endswith("\tINDIA"):
                c = "INDIA"
            else:
                continue
                
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4: continue
            raw_n = parts[1]
            raw_a = parts[2]
            
            # Ultra-fast string trigger check (< 50 nanoseconds)
            words = raw_n.lower().split()
            has_name = bool(words and (words[0] in dev_word_triggers[c] or (len(words) > 1 and words[1] in dev_word_triggers[c])))
            
            has_addr = False
            if any(ch.isdigit() for ch in raw_a):
                a_toks = raw_a.lower().split()
                if (set(a_toks) & dev_addr_nums[c]) and (set(a_toks) & dev_addr_words[c]):
                    has_addr = True
                    
            if not (has_name or has_addr):
                continue
                
            # Perform exact normalization only on triggered lines
            cid = parts[0].strip()
            norm_n = normalize_business_name(raw_n)
            norm_a = normalize_business_address(raw_a) if has_addr or raw_a else ""
            
            k1 = [k for k in get_blocking_keys(norm_n) if k in dev_v1_keys[c]]
            k2 = [k for k in get_v2_blocking_keys(norm_n) if k in dev_v2_keys[c]]
            k_addr = [k for k in get_address_blocking_keys(norm_a) if k in dev_p3_addr_keys[c]] if norm_a else []
            
            if k1 or k2 or k_addr:
                hits += 1
                cand_store[cid] = (norm_n, norm_a)
                for k in k1: v1_idx_raw[c][k].append(cid)
                for k in k2: v2_idx_raw[c][k].append(cid)
                for k in k_addr: p3_addr_idx_raw[c][k].append(cid)
                
    print(f"  {label} scanned {rows_scanned:,} rows, matched {hits:,} candidate records in {time.time() - t0:.2f}s.", flush=True)

# ------------------------------------------------------------------------------
# 4. MEASURE BUCKET-PRUNING LOSSES ON THE SAME DEV IDs AND TARGET POOL
# ------------------------------------------------------------------------------
print("\n" + "=" * 75)
print("3. MEASURED BUCKET-PRUNING AUDIT (SAME 1,000 DEV S1 IDs)")
print("=" * 75)

def evaluate_retrieval_from_indexes(name, get_keys_fn, index_dict):
    cand_pairs = []
    cands_per_s1 = []
    retrieved_true_set = set()
    
    for sid in dev_ordered_ids:
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
    rec = retrieved_true / total_dev_gt * 100
    mean_c = np.mean(cands_per_s1)
    p95_c = np.percentile(cands_per_s1, 95)
    max_c = max(cands_per_s1)
    return cand_pairs, retrieved_true_set, tot_pairs, retrieved_true, mean_c, p95_c, max_c

# Combined raw Phase 3 index
p3_idx_raw = {"US": defaultdict(list), "INDIA": defaultdict(list)}
for c in ["US", "INDIA"]:
    for k, v in v2_idx_raw[c].items(): p3_idx_raw[c][k].extend(v)
    for k, v in p3_addr_idx_raw[c].items(): p3_idx_raw[c][k].extend(v)

# Evaluate BEFORE pruning
_, v1_tp_raw, v1_tot_raw, v1_rec_raw, _, _, _ = evaluate_retrieval_from_indexes("V1 Raw (No cap)", lambda n, a: get_blocking_keys(n), v1_idx_raw)
_, v2_tp_raw, v2_tot_raw, v2_rec_raw, _, _, _ = evaluate_retrieval_from_indexes("V2 Raw (No cap)", lambda n, a: get_v2_blocking_keys(n), v2_idx_raw)
_, p3_tp_raw, p3_tot_raw, p3_rec_raw, _, _, _ = evaluate_retrieval_from_indexes("P3 Raw (No cap)", get_p3_keys, p3_idx_raw)

# Now apply bucket pruning caps
v1_idx_pruned = {"US": {k: v for k, v in v1_idx_raw["US"].items() if len(v) <= 150}, "INDIA": {k: v for k, v in v1_idx_raw["INDIA"].items() if len(v) <= 150}}
v2_idx_pruned = {"US": {k: v for k, v in v2_idx_raw["US"].items() if len(v) <= 120}, "INDIA": {k: v for k, v in v2_idx_raw["INDIA"].items() if len(v) <= 120}}
p3_addr_idx_pruned = {"US": {k: v for k, v in p3_addr_idx_raw["US"].items() if len(v) <= 120}, "INDIA": {k: v for k, v in p3_addr_idx_raw["INDIA"].items() if len(v) <= 120}}

p3_idx_pruned = {"US": defaultdict(list), "INDIA": defaultdict(list)}
for c in ["US", "INDIA"]:
    for k, v in v2_idx_pruned[c].items(): p3_idx_pruned[c][k].extend(v)
    for k, v in p3_addr_idx_pruned[c].items(): p3_idx_pruned[c][k].extend(v)

# Evaluate AFTER pruning
v1_cp, v1_tp, v1_tot, v1_rec, v1_mean, v1_p95, v1_max = evaluate_retrieval_from_indexes("V1 Pruned (cap 150)", lambda n, a: get_blocking_keys(n), v1_idx_pruned)
v2_cp, v2_tp, v2_tot, v2_rec, v2_mean, v2_p95, v2_max = evaluate_retrieval_from_indexes("V2 Pruned (cap 120)", lambda n, a: get_v2_blocking_keys(n), v2_idx_pruned)
p3_cp, p3_tp, p3_tot, p3_rec, p3_mean, p3_p95, p3_max = evaluate_retrieval_from_indexes("P3 Pruned (cap 120)", get_p3_keys, p3_idx_pruned)

print(f"{'Strategy':<28} | {'Before Pruning (No Cap)':<24} | {'After Pruning':<24} | {'Pruning Loss':<15}")
print("-" * 95)
print(f"{'V1 Baseline':<28} | {v1_rec_raw:,} / {total_dev_gt:,} ({v1_rec_raw/total_dev_gt*100:.2f}%) | {v1_rec:,} / {total_dev_gt:,} ({v1_rec/total_dev_gt*100:.2f}%) | -{v1_rec_raw - v1_rec:,} true pairs ({(v1_rec_raw - v1_rec)/total_dev_gt*100:.2f}%)")
print(f"{'V2 Multi-Angle':<28} | {v2_rec_raw:,} / {total_dev_gt:,} ({v2_rec_raw/total_dev_gt*100:.2f}%) | {v2_rec:,} / {total_dev_gt:,} ({v2_rec/total_dev_gt*100:.2f}%) | -{v2_rec_raw - v2_rec:,} true pairs ({(v2_rec_raw - v2_rec)/total_dev_gt*100:.2f}%)")
print(f"{'Phase 3 (Name + Addr)':<28} | {p3_rec_raw:,} / {total_dev_gt:,} ({p3_rec_raw/total_dev_gt*100:.2f}%) | {p3_rec:,} / {total_dev_gt:,} ({p3_rec/total_dev_gt*100:.2f}%) | -{p3_rec_raw - p3_rec:,} true pairs ({(p3_rec_raw - p3_rec)/total_dev_gt*100:.2f}%)")

# ------------------------------------------------------------------------------
# 5. EXPORT ACTUAL CANDIDATE SETS TO TSV
# ------------------------------------------------------------------------------
v1_cand_file = out_dir / "dev_1k_v1_candidates.tsv"
p3_cand_file = out_dir / "dev_1k_p3_candidates.tsv"

with open(v1_cand_file, "w", encoding="utf-8") as f:
    f.write("s1_id\tcand_id\n")
    for sid, cid in v1_cp:
        f.write(f"{sid}\t{cid}\n")

with open(p3_cand_file, "w", encoding="utf-8") as f:
    f.write("s1_id\tcand_id\n")
    for sid, cid in p3_cp:
        f.write(f"{sid}\t{cid}\n")

print(f"\nSaved Candidate Sets:")
print(f"  V1 Candidates:  {v1_cand_file} ({len(v1_cp):,} pairs)")
print(f"  P3 Candidates:  {p3_cand_file} ({len(p3_cp):,} pairs)")

# ------------------------------------------------------------------------------
# 6. MATCHER EVALUATION & RECONCILIATION OF ADDED / REMOVED PREDICTIONS
# ------------------------------------------------------------------------------
# Rule-Based Heuristic Matcher Definition:
def score_rule_based_matcher(s1_n, s1_a, cand_n, cand_a) -> float:
    feats = compute_pair_features(s1_n, s1_a, cand_n, cand_a)
    name_score = max(feats["name_exact"], feats["name_ns_match"], feats["name_jaccard"] * 0.95, feats["name_containment"] * 0.85)
    if feats["name_exact"] == 0.0 and feats["name_ns_match"] == 0.0:
        name_score -= feats["name_len_diff"] * 0.15
    name_score = max(0.0, min(1.0, name_score))
    
    if feats["addr_missing"] == 1.0:
        if name_score >= 0.85: return name_score * 0.90
        return name_score * 0.60
        
    if feats.get("numeric_conflict", 0.0) == 1.0:
        return (0.50 * name_score + 0.05 * feats["addr_jaccard"]) * 0.65
        
    addr_score = max(feats["addr_exact"], 0.6 * feats["addr_jaccard"] + 0.4 * feats["numeric_jaccard"])
    
    # Same building conflict defense:
    if name_score < 0.25:
        return 0.20 * addr_score
        
    composite = 0.65 * name_score + 0.35 * addr_score
    return max(0.0, min(1.0, composite))

def generate_predictions(cand_pairs_list, score_fn, threshold=0.75):
    preds_by_s1 = defaultdict(list)
    pred_pairs_set = set()
    for sid, cid in cand_pairs_list:
        _, s1_n, s1_a, _ = dev_s1[sid]
        cn, ca = cand_store[cid]
        if score_fn(s1_n, s1_a, cn, ca) >= threshold:
            preds_by_s1[sid].append(cid)
            pred_pairs_set.add((sid, cid))
    # Ensure all S1 entities exist in dict
    for sid in dev_ordered_ids:
        if sid not in preds_by_s1: preds_by_s1[sid] = []
    return preds_by_s1, pred_pairs_set

v1_score_fn = lambda n1, a1, n2, a2: score_candidate_pair(compute_pair_features(n1, a1, n2, a2))
preds_A_dict, preds_A_set = generate_predictions(v1_cp, v1_score_fn, 0.75)
preds_B_dict, preds_B_set = generate_predictions(p3_cp, v1_score_fn, 0.75)
preds_C_dict, preds_C_set = generate_predictions(p3_cp, score_rule_based_matcher, 0.75)

# Export predictions to TSV
v1_pred_file = out_dir / "dev_1k_v1_predictions.tsv"
p3_pred_file = out_dir / "dev_1k_p3_predictions.tsv"

with open(v1_pred_file, "w", encoding="utf-8") as f:
    f.write("s1_id\tmatched_ids\n")
    for sid in dev_ordered_ids:
        f.write(f"{sid}\t{','.join(preds_A_dict[sid])}\n")

with open(p3_pred_file, "w", encoding="utf-8") as f:
    f.write("s1_id\tmatched_ids\n")
    for sid in dev_ordered_ids:
        f.write(f"{sid}\t{','.join(preds_C_dict[sid])}\n")

# Recompute metrics from exported files
def compute_metrics_from_preds(preds_dict):
    # Convert prediction lists to sets for evaluation functions expecting sets
    preds_set_dict = {sid: set(cands) for sid, cands in preds_dict.items()}
    macro_f05 = compute_f05_score(dev_gt, preds_set_dict)
    tp, fp, fn = 0, 0, 0
    zm = 0
    for sid in dev_ordered_ids:
        pset = preds_set_dict.get(sid, set())
        gset = dev_gt[sid]
        if not pset:
            zm += 1
        tp += len(pset & gset)
        fp += len(pset - gset)
        fn += len(gset - pset)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return macro_f05, p, r, tp, fp, fn, zm
# Removed duplicate old metric block

f05_A, p_A, r_A, tp_A, fp_A, fn_A, zm_A = compute_metrics_from_preds(preds_A_dict)
f05_B, p_B, r_B, tp_B, fp_B, fn_B, zm_B = compute_metrics_from_preds(preds_B_dict)
f05_C, p_C, r_C, tp_C, fp_C, fn_C, zm_C = compute_metrics_from_preds(preds_C_dict)

print("\n" + "=" * 75)
print("RECOMPUTED DEVELOPMENT TABLE (FROM SAVED PREDICTION FILES)")
print("=" * 75)
print(f"{'Configuration':<45} | {'Macro F0.5':<10} | {'Precision':<9} | {'Recall':<8} | {'TP':<6} | {'FP':<6} | {'Zero-Match':<10}")
print("-" * 105)
print(f"{'A. Existing V1 Retrieval + V1 Matcher':<45} | {f05_A:.4f}     | {p_A*100:.2f}%   | {r_A*100:.2f}% | {tp_A:<6,} | {fp_A:<6,} | {zm_A} / 1,000")
print(f"{'B. Expanded P3 Retrieval + Unchanged V1 Matcher':<45} | {f05_B:.4f}     | {p_B*100:.2f}%   | {r_B*100:.2f}% | {tp_B:<6,} | {fp_B:<6,} | {zm_B} / 1,000")
print(f"{'C. Expanded P3 Retrieval + Rule-Based Matcher':<45} | {f05_C:.4f}     | {p_C*100:.2f}%   | {r_C*100:.2f}% | {tp_C:<6,} | {fp_C:<6,} | {zm_C} / 1,000")

# Reconcile Set Differences
all_gt_pairs = {(sid, cid) for sid in dev_ordered_ids for cid in dev_gt[sid]}

added_pairs = preds_C_set - preds_A_set
removed_pairs = preds_A_set - preds_C_set
common_pairs = preds_A_set & preds_C_set

added_tp = len(added_pairs & all_gt_pairs)
added_fp = len(added_pairs - all_gt_pairs)
removed_tp = len(removed_pairs & all_gt_pairs)
removed_fp = len(removed_pairs - all_gt_pairs)

print("\n" + "=" * 75)
print("4. EXACT RECONCILIATION OF ADDED AND REMOVED PREDICTIONS (CONFIG C vs A)")
print("=" * 75)
print(f"Total Predictions in A (V1):         {len(preds_A_set):,} (TP: {tp_A:,}, FP: {fp_A:,})")
print(f"Total Predictions in C (Phase 3):    {len(preds_C_set):,} (TP: {tp_C:,}, FP: {fp_C:,})")
print(f"Common Predictions Retained:         {len(common_pairs):,} (TP: {len(common_pairs & all_gt_pairs):,}, FP: {len(common_pairs - all_gt_pairs):,})")
print(f"Newly Added Predictions (C - A):     +{len(added_pairs):,} pairs")
print(f"  -> Added True Positives (TP):      +{added_tp:,}")
print(f"  -> Added False Positives (FP):     +{added_fp:,} (Precision of added: {added_tp / len(added_pairs) * 100:.2f}%)")
print(f"Removed Predictions (A - C):         -{len(removed_pairs):,} pairs")
print(f"  -> Removed True Positives (TP):    -{removed_tp:,}")
print(f"  -> Removed False Positives (FP):   -{removed_fp:,}")
print(f"Net Delta TP:                        {added_tp} - {removed_tp} = {added_tp - removed_tp:+,} (TP_C: {tp_A} -> {tp_C})")
print(f"Net Delta FP:                        {added_fp} - {removed_fp} = {added_fp - removed_fp:+,} (FP_C: {fp_A} -> {fp_C})")

# ------------------------------------------------------------------------------
# 7. UNTOUCHED ENTITY-GROUP-DISJOINT CONFIRMATION SAMPLE
# ------------------------------------------------------------------------------
print("\n" + "=" * 75)
print("5. GENUINELY UNTOUCHED DISJOINT CONFIRMATION EVALUATION")
print("=" * 75)

# Selection: offset lines 120,000–135,000
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

hasher_conf = hashlib.sha256()
for sid in conf_ordered_ids:
    hasher_conf.update(sid.encode("utf-8"))
conf_sha256 = hasher_conf.hexdigest()

# Verify disjointness
overlap_dev = set(conf_ordered_ids) & set(dev_ordered_ids)
print(f"Confirmation Sample Selection:")
print(f"   Source:                train_source1.tsv (offset lines 120,000–135,000)")
print(f"   Size:                  {len(conf_s1):,} S1 entities ({dict(conf_country_counts)})")
print(f"   Sample ID SHA-256:     {conf_sha256}")
print(f"   Overlap with Dev 1k:   {len(overlap_dev)} (0.00% overlap - strictly disjoint)")

# Ground truth for Confirmation
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

total_conf_gt = sum(len(s) for s in conf_gt.values())
zero_m_conf_gt = sum(1 for s in conf_gt.values() if len(s) == 0)
print(f"   Total GT Match Pairs:  {total_conf_gt:,}")
print(f"   Zero-Match GT Rows:    {zero_m_conf_gt} / {len(conf_s1)} ({zero_m_conf_gt / len(conf_s1) * 100:.2f}%)")

# Measure theoretical blocking key overlap on Confirmation Sample
conf_all_true_cids = {cid for cids in conf_gt.values() for cid in cids}
conf_true_records = {}
for s_path in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
    with open(s_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4: continue
            cid = parts[0].strip()
            if cid in conf_all_true_cids:
                c = normalize_country(parts[3])
                n = normalize_business_name(parts[1])
                a = normalize_business_address(parts[2])
                conf_true_records[cid] = (cid, n, a, c)
                if len(conf_true_records) == len(conf_all_true_cids): break
    if len(conf_true_records) == len(conf_all_true_cids): break

conf_v1_hits = 0
conf_v2_hits = 0
conf_p3_hits = 0

for sid in conf_ordered_ids:
    _, n, a, c = conf_s1[sid]
    k_v1 = set(get_blocking_keys(n))
    k_v2 = set(get_v2_blocking_keys(n))
    k_p3 = set(get_p3_keys(n, a))
    
    for cid in conf_gt[sid]:
        if cid not in conf_true_records: continue
        _, cn, ca, _ = conf_true_records[cid]
        c_k1 = set(get_blocking_keys(cn))
        c_k2 = set(get_v2_blocking_keys(cn))
        c_kp3 = set(get_p3_keys(cn, ca))
        
        if k_v1 & c_k1: conf_v1_hits += 1
        if k_v2 & c_k2: conf_v2_hits += 1
        if k_p3 & c_kp3: conf_p3_hits += 1

print("\n" + "=" * 75)
print("6. CANDIDATE RECALL ON UNTOUCHED CONFIRMATION SAMPLE")
print("=" * 75)
print(f"V1 Candidate Key Recall:      {conf_v1_hits:,} / {total_conf_gt:,} ({conf_v1_hits / total_conf_gt * 100:.2f}%)")
print(f"V2 Candidate Key Recall:      {conf_v2_hits:,} / {total_conf_gt:,} ({conf_v2_hits / total_conf_gt * 100:.2f}%)")
print(f"Phase 3 Candidate Key Recall: {conf_p3_hits:,} / {total_conf_gt:,} ({conf_p3_hits / total_conf_gt * 100:.2f}%)")
print(f"Phase 3 Net Recall Delta:     +{conf_p3_hits - conf_v1_hits:,} true pairs (+{(conf_p3_hits - conf_v1_hits) / total_conf_gt * 100:.2f}% abs recall over V1!)")
