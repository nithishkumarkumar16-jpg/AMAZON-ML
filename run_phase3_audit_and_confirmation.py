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

print("=" * 75, flush=True)
print("PHASE 3: BUCKET-PRUNING AUDIT & UNTOUCHED CONFIRMATION BENCHMARK", flush=True)
print("=" * 75, flush=True)

# --- Address Blocking Key Generator ---
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

# --- Matcher Definitions ---
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


# ------------------------------------------------------------------------------
# 1. SAMPLE SELECTION (FAST, FILTERED BY LINE NUMBER FIRST)
# ------------------------------------------------------------------------------
dev_s1 = {}
dev_ordered_ids = []
dev_country_counts = Counter()

conf_s1 = {}
conf_ordered_ids = []
conf_country_counts = Counter()

t0_sample = time.time()
with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
    f.readline()
    for idx, line in enumerate(f):
        # Fast filter by line offset
        is_dev_range = (20000 <= idx < 25000 and len(dev_s1) < 1000)
        is_conf_range = (120000 <= idx < 135000 and len(conf_s1) < 1000)
        if not (is_dev_range or is_conf_range):
            if len(dev_s1) == 1000 and len(conf_s1) == 1000:
                break
            continue
            
        parts = line.rstrip("\r\n").split("\t")
        if len(parts) < 4: continue
        sid = parts[0].strip()
        c = normalize_country(parts[3])
        if c not in {"US", "INDIA"}: continue
        n = normalize_business_name(parts[1])
        a = normalize_business_address(parts[2])
        
        if is_dev_range:
            dev_s1[sid] = (sid, n, a, c)
            dev_ordered_ids.append(sid)
            dev_country_counts[c] += 1
            
        elif is_conf_range:
            conf_s1[sid] = (sid, n, a, c)
            conf_ordered_ids.append(sid)
            conf_country_counts[c] += 1

dev_sha = hashlib.sha256("".join(dev_ordered_ids).encode("utf-8")).hexdigest()
conf_sha = hashlib.sha256("".join(conf_ordered_ids).encode("utf-8")).hexdigest()
overlap = set(dev_ordered_ids) & set(conf_ordered_ids)

print(f"Sample loading took {time.time() - t0_sample:.2f}s", flush=True)
print(f"Dev Sample (1,000 S1):          SHA-256 = {dev_sha} | Countries = {dict(dev_country_counts)}", flush=True)
print(f"Confirmation Sample (1,000 S1):  SHA-256 = {conf_sha} | Countries = {dict(conf_country_counts)}", flush=True)
print(f"Overlap between Dev & Conf:    {len(overlap)} entities (0.00% overlap - strictly disjoint)", flush=True)

# Load Ground Truth
dev_gt = {}
conf_gt = {}
all_target_sids = set(dev_ordered_ids) | set(conf_ordered_ids)

with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        parts = line.rstrip("\r\n").split("\t")
        sid = parts[0].strip()
        if sid in all_target_sids:
            m_val = parts[1] if len(parts) > 1 else ""
            m_ids = {m.strip() for m in m_val.split(",") if m.strip()} if m_val else set()
            if sid in dev_s1: dev_gt[sid] = m_ids
            if sid in conf_s1: conf_gt[sid] = m_ids
            if len(dev_gt) == len(dev_s1) and len(conf_gt) == len(conf_s1):
                break

dev_tot_gt = sum(len(s) for s in dev_gt.values())
dev_zm_gt = sum(1 for s in dev_gt.values() if len(s) == 0)

conf_tot_gt = sum(len(s) for s in conf_gt.values())
conf_zm_gt = sum(1 for s in conf_gt.values() if len(s) == 0)

print(f"\nGround Truth Summary:", flush=True)
print(f"  Dev 1k:   {dev_tot_gt:,} true match pairs | Zero-match entities: {dev_zm_gt} / {len(dev_s1)} ({dev_zm_gt/len(dev_s1)*100:.2f}%)", flush=True)
print(f"  Conf 1k:  {conf_tot_gt:,} true match pairs | Zero-match entities: {conf_zm_gt} / {len(conf_s1)} ({conf_zm_gt/len(conf_s1)*100:.2f}%)", flush=True)


# ------------------------------------------------------------------------------
# 2. BUILD TARGET BLOCKING KEYS & STREAM S2/S3
# ------------------------------------------------------------------------------
dev_v1_keys = {"US": set(), "INDIA": set()}
dev_v2_keys = {"US": set(), "INDIA": set()}
dev_p3_addr_keys = {"US": set(), "INDIA": set()}

for sid, (_, n, a, c) in dev_s1.items():
    dev_v1_keys[c].update(get_blocking_keys(n))
    dev_v2_keys[c].update(get_v2_blocking_keys(n))
    dev_p3_addr_keys[c].update(get_address_blocking_keys(a))

conf_v1_keys = {"US": set(), "INDIA": set()}
conf_v2_keys = {"US": set(), "INDIA": set()}
conf_p3_addr_keys = {"US": set(), "INDIA": set()}

for sid, (_, n, a, c) in conf_s1.items():
    conf_v1_keys[c].update(get_blocking_keys(n))
    conf_v2_keys[c].update(get_v2_blocking_keys(n))
    conf_p3_addr_keys[c].update(get_address_blocking_keys(a))

all_name_keys = {"US": set(), "INDIA": set()}
all_addr_keys = {"US": set(), "INDIA": set()}
for c in ["US", "INDIA"]:
    all_name_keys[c] = dev_v1_keys[c] | dev_v2_keys[c] | conf_v1_keys[c] | conf_v2_keys[c]
    all_addr_keys[c] = dev_p3_addr_keys[c] | conf_p3_addr_keys[c]

dev_v1_idx = {"US": defaultdict(list), "INDIA": defaultdict(list)}
dev_v2_idx = {"US": defaultdict(list), "INDIA": defaultdict(list)}
dev_p3_addr_idx = {"US": defaultdict(list), "INDIA": defaultdict(list)}

conf_v1_idx = {"US": defaultdict(list), "INDIA": defaultdict(list)}
conf_v2_idx = {"US": defaultdict(list), "INDIA": defaultdict(list)}
conf_p3_addr_idx = {"US": defaultdict(list), "INDIA": defaultdict(list)}

cand_store = {}

print(f"\nStreaming S2 and S3 against full permitted target pool...", flush=True)
t_start = time.time()
for spath, label in [("dataset/train/train_source2.tsv", "S2"), ("dataset/train/train_source3.tsv", "S3")]:
    t0 = time.time()
    rows = 0
    with open(spath, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            rows += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4: continue
            raw_c = parts[3].strip()
            c = normalize_country(raw_c)
            if c not in {"US", "INDIA"}: continue
            
            cid = parts[0].strip()
            raw_name = parts[1]
            raw_addr = parts[2]
            
            norm_n = normalize_business_name(raw_name)
            
            k1 = [k for k in get_blocking_keys(norm_n) if k in all_name_keys[c]]
            k2 = [k for k in get_v2_blocking_keys(norm_n) if k in all_name_keys[c]]
            
            k_addr = []
            if any(ch.isdigit() for ch in raw_addr):
                norm_a = normalize_business_address(raw_addr)
                k_addr = [k for k in get_address_blocking_keys(norm_a) if k in all_addr_keys[c]]
            else:
                norm_a = ""
                
            if k1 or k2 or k_addr:
                if not norm_a and raw_addr:
                    norm_a = normalize_business_address(raw_addr)
                cand_store[cid] = (norm_n, norm_a)
                
                # Dev indexing
                for k in k1:
                    if k in dev_v1_keys[c]: dev_v1_idx[c][k].append(cid)
                    if k in conf_v1_keys[c]: conf_v1_idx[c][k].append(cid)
                for k in k2:
                    if k in dev_v2_keys[c]: dev_v2_idx[c][k].append(cid)
                    if k in conf_v2_keys[c]: conf_v2_idx[c][k].append(cid)
                for k in k_addr:
                    if k in dev_p3_addr_keys[c]: dev_p3_addr_idx[c][k].append(cid)
                    if k in conf_p3_addr_keys[c]: conf_p3_addr_idx[c][k].append(cid)
                    
    print(f"  {label} streamed {rows:,} rows in {time.time() - t0:.1f}s", flush=True)

print(f"Streaming complete in {time.time() - t_start:.1f}s. Distinct candidates indexed: {len(cand_store):,}", flush=True)


# ------------------------------------------------------------------------------
# 3. MEASURE BUCKET-PRUNING LOSSES (BEFORE & AFTER PRUNING ON SAME DEV 1K)
# ------------------------------------------------------------------------------
print("\n" + "=" * 75, flush=True)
print("BUCKET-PRUNING LOSS MEASUREMENT ON SAME DEV 1K ENTITIES", flush=True)
print("=" * 75, flush=True)

def evaluate_retrieval_loss(s1_dict, gt_dict, ordered_ids, key_fn, index_dict):
    pairs = 0
    true_ret = 0
    for sid in ordered_ids:
        _, n, a, c = s1_dict[sid]
        keys = key_fn(n, a)
        c_set = set()
        c_idx = index_dict[c]
        for k in keys:
            if k in c_idx:
                c_set.update(c_idx[k])
        pairs += len(c_set)
        true_ret += len(c_set & gt_dict[sid])
    return pairs, true_ret

# Before pruning
v1_b_pairs, v1_b_true = evaluate_retrieval_loss(dev_s1, dev_gt, dev_ordered_ids, lambda n, a: get_blocking_keys(n), dev_v1_idx)
v2_b_pairs, v2_b_true = evaluate_retrieval_loss(dev_s1, dev_gt, dev_ordered_ids, lambda n, a: get_v2_blocking_keys(n), dev_v2_idx)

dev_p3_idx_before = {"US": defaultdict(list), "INDIA": defaultdict(list)}
for c in ["US", "INDIA"]:
    for k, v in dev_v2_idx[c].items(): dev_p3_idx_before[c][k].extend(v)
    for k, v in dev_p3_addr_idx[c].items(): dev_p3_idx_before[c][k].extend(v)

p3_b_pairs, p3_b_true = evaluate_retrieval_loss(dev_s1, dev_gt, dev_ordered_ids, get_p3_keys, dev_p3_idx_before)

# Apply Bucket Caps
MAX_BUCKET_V1 = 150
MAX_BUCKET_V2 = 120
MAX_BUCKET_P3 = 120

dev_v1_pruned = {"US": {k: v for k, v in dev_v1_idx["US"].items() if len(v) <= MAX_BUCKET_V1},
                 "INDIA": {k: v for k, v in dev_v1_idx["INDIA"].items() if len(v) <= MAX_BUCKET_V1}}

dev_v2_pruned = {"US": {k: v for k, v in dev_v2_idx["US"].items() if len(v) <= MAX_BUCKET_V2},
                 "INDIA": {k: v for k, v in dev_v2_idx["INDIA"].items() if len(v) <= MAX_BUCKET_V2}}

dev_p3_addr_pruned = {"US": {k: v for k, v in dev_p3_addr_idx["US"].items() if len(v) <= MAX_BUCKET_P3},
                      "INDIA": {k: v for k, v in dev_p3_addr_idx["INDIA"].items() if len(v) <= MAX_BUCKET_P3}}

dev_p3_idx_pruned = {"US": defaultdict(list), "INDIA": defaultdict(list)}
for c in ["US", "INDIA"]:
    for k, v in dev_v2_pruned[c].items(): dev_p3_idx_pruned[c][k].extend(v)
    for k, v in dev_p3_addr_pruned[c].items(): dev_p3_idx_pruned[c][k].extend(v)

# After pruning
v1_a_pairs, v1_a_true = evaluate_retrieval_loss(dev_s1, dev_gt, dev_ordered_ids, lambda n, a: get_blocking_keys(n), dev_v1_pruned)
v2_a_pairs, v2_a_true = evaluate_retrieval_loss(dev_s1, dev_gt, dev_ordered_ids, lambda n, a: get_v2_blocking_keys(n), dev_v2_pruned)
p3_a_pairs, p3_a_true = evaluate_retrieval_loss(dev_s1, dev_gt, dev_ordered_ids, get_p3_keys, dev_p3_idx_pruned)

print(f"{'Strategy':<20} | {'Before Pruning (True / Rec)':<30} | {'After Pruning (True / Rec)':<30} | {'Pruning Loss':<18}", flush=True)
print("-" * 105, flush=True)
print(f"{'V1 (Cap 150)':<20} | {v1_b_true:,} / {dev_tot_gt:,} ({v1_b_true/dev_tot_gt*100:.2f}%)       | {v1_a_true:,} / {dev_tot_gt:,} ({v1_a_true/dev_tot_gt*100:.2f}%)       | -{v1_b_true - v1_a_true:,} (-{(v1_b_true - v1_a_true)/dev_tot_gt*100:.2f}%)", flush=True)
print(f"{'V2 (Cap 120)':<20} | {v2_b_true:,} / {dev_tot_gt:,} ({v2_b_true/dev_tot_gt*100:.2f}%)       | {v2_a_true:,} / {dev_tot_gt:,} ({v2_a_true/dev_tot_gt*100:.2f}%)       | -{v2_b_true - v2_a_true:,} (-{(v2_b_true - v2_a_true)/dev_tot_gt*100:.2f}%)", flush=True)
print(f"{'Phase 3 (Cap 120)':<20} | {p3_b_true:,} / {dev_tot_gt:,} ({p3_b_true/dev_tot_gt*100:.2f}%)       | {p3_a_true:,} / {dev_tot_gt:,} ({p3_a_true/dev_tot_gt*100:.2f}%)       | -{p3_b_true - p3_a_true:,} (-{(p3_b_true - p3_a_true)/dev_tot_gt*100:.2f}%)", flush=True)
print("-" * 105, flush=True)
print(f"Key Finding: Phase 3 retains +{p3_a_true - v1_a_true:,} true pairs (+{(p3_a_true - v1_a_true)/dev_tot_gt*100:.2f}% recall) over V1 after pruning.", flush=True)


# ------------------------------------------------------------------------------
# 4. CONFIRMATION BENCHMARK (UNTOUCHED 1,000 S1 ENTITIES)
# ------------------------------------------------------------------------------
print("\n" + "=" * 75, flush=True)
print("CONFIRMATION BENCHMARK ON UNTOUCHED 1,000 S1 ENTITIES", flush=True)
print("=" * 75, flush=True)

# Prune confirmation index
conf_v1_pruned = {"US": {k: v for k, v in conf_v1_idx["US"].items() if len(v) <= MAX_BUCKET_V1},
                  "INDIA": {k: v for k, v in conf_v1_idx["INDIA"].items() if len(v) <= MAX_BUCKET_V1}}

conf_v2_pruned = {"US": {k: v for k, v in conf_v2_idx["US"].items() if len(v) <= MAX_BUCKET_V2},
                  "INDIA": {k: v for k, v in conf_v2_idx["INDIA"].items() if len(v) <= MAX_BUCKET_V2}}

conf_p3_addr_pruned = {"US": {k: v for k, v in conf_p3_addr_idx["US"].items() if len(v) <= MAX_BUCKET_P3},
                       "INDIA": {k: v for k, v in conf_p3_addr_idx["INDIA"].items() if len(v) <= MAX_BUCKET_P3}}

conf_p3_idx_pruned = {"US": defaultdict(list), "INDIA": defaultdict(list)}
for c in ["US", "INDIA"]:
    for k, v in conf_v2_pruned[c].items(): conf_p3_idx_pruned[c][k].extend(v)
    for k, v in conf_p3_addr_pruned[c].items(): conf_p3_idx_pruned[c][k].extend(v)

def run_eval_pipeline(s1_dict, gt_dict, ordered_ids, key_fn, index_dict, score_fn, threshold):
    cand_pairs_count = 0
    retrieved_true = 0
    preds_dict = defaultdict(list)
    s1_all = {sid: [] for sid in ordered_ids}
    
    for sid in ordered_ids:
        _, s1_n, s1_a, c = s1_dict[sid]
        keys = key_fn(s1_n, s1_a)
        c_set = set()
        c_idx = index_dict[c]
        for k in keys:
            if k in c_idx:
                c_set.update(c_idx[k])
        cand_pairs_count += len(c_set)
        retrieved_true += len(c_set & gt_dict[sid])
        
        for cid in c_set:
            cn, ca = cand_store[cid]
            sc = score_fn(s1_n, s1_a, cn, ca)
            if sc >= threshold:
                s1_all[sid].append(cid)
                preds_dict[sid].append(cid)
                
    f05 = compute_f05_score(gt_dict, {sid: set(v) for sid, v in s1_all.items()})
    
    tp, fp, fn = 0, 0, 0
    zm = 0
    for sid in ordered_ids:
        pset = set(s1_all[sid])
        gset = gt_dict[sid]
        if not pset: zm += 1
        tp += len(pset & gset)
        fp += len(pset - gset)
        fn += len(gset - pset)
        
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tot_gt = sum(len(s) for s in gt_dict.values())
    c_rec = retrieved_true / tot_gt * 100
    
    return f05, p, r, c_rec, tp, fp, fn, zm, cand_pairs_count, retrieved_true, preds_dict

f05_conf_v1, p_conf_v1, r_conf_v1, cr_conf_v1, tp_conf_v1, fp_conf_v1, fn_conf_v1, zm_conf_v1, cp_conf_v1, rt_conf_v1, preds_conf_v1 = run_eval_pipeline(
    conf_s1, conf_gt, conf_ordered_ids,
    lambda n, a: get_blocking_keys(n),
    conf_v1_pruned,
    score_v1,
    0.75
)

f05_conf_p3, p_conf_p3, r_conf_p3, cr_conf_p3, tp_conf_p3, fp_conf_p3, fn_conf_p3, zm_conf_p3, cp_conf_p3, rt_conf_p3, preds_conf_p3 = run_eval_pipeline(
    conf_s1, conf_gt, conf_ordered_ids,
    get_p3_keys,
    conf_p3_idx_pruned,
    score_p3_rule_based,
    0.75
)

print(f"{'Configuration':<35} | {'Macro F0.5':<10} | {'Precision':<9} | {'Recall':<8} | {'Cand Rec':<9} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Zero-Match':<10}", flush=True)
print("-" * 115, flush=True)
print(f"{'V1 Baseline (Name, Matcher @ 0.75)':<35} | {f05_conf_v1:.4f}     | {p_conf_v1*100:.2f}%   | {r_conf_v1*100:.2f}% | {cr_conf_v1:.2f}%    | {tp_conf_v1:<6} | {fp_conf_v1:<6} | {fn_conf_v1:<6} | {zm_conf_v1} / {len(conf_s1)}", flush=True)
print(f"{'Phase 3 (Name+Addr, Rule-Based @ 0.75)':<35} | {f05_conf_p3:.4f}     | {p_conf_p3*100:.2f}%   | {r_conf_p3*100:.2f}% | {cr_conf_p3:.2f}%    | {tp_conf_p3:<6} | {fp_conf_p3:<6} | {fn_conf_p3:<6} | {zm_conf_p3} / {len(conf_s1)}", flush=True)
print("-" * 115, flush=True)
print(f"Delta Macro F0.5:   {f05_conf_p3 - f05_conf_v1:+.4f}", flush=True)
print(f"Delta Precision:    {(p_conf_p3 - p_conf_v1)*100:+.2f}%", flush=True)
print(f"Delta Recall:       {(r_conf_p3 - r_conf_v1)*100:+.2f}%", flush=True)
print(f"Delta Cand Recall:  {(cr_conf_p3 - cr_conf_v1):+.2f}% (+{rt_conf_p3 - rt_conf_v1} true pairs retrieved)", flush=True)
