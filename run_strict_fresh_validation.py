"""Strict Fresh Validation Benchmark.

Imports score_pair_v3 DIRECTLY from predict_v3 (guaranteeing one shared scoring function).
Samples 3,000 entities from train_source1 rows 10,000-13,000 (0% overlap with original 5k).
Verifies SHA-256 checksum of sample IDs and prints exact country counts.
Evaluates:
  1. V1 Matcher @ 0.75
  2. V3 Matcher (score_pair_v3) @ 0.75
"""

import hashlib
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

src_dir = Path(__file__).resolve().parent / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(src_dir))

# DIRECT IMPORT FROM PRODUCTION MODULE
from predict_v3 import score_pair_v3, SELECTED_THRESHOLD
from normalization import normalize_business_name, normalize_business_address, normalize_country
from features import compute_pair_features, score_candidate_pair
from evaluation import compute_f05_score
from blocking import get_blocking_keys
from predict_v2 import get_v2_blocking_keys

# Load original 5k IDs to prove 0% overlap
with open("output/val_cache_5k.pkl", "rb") as f:
    orig_s1, orig_gt, _ = pickle.load(f)
orig_5k_ids = set(orig_s1.keys())

# Extract fresh 3,000 S1 entities from rows 10,000 to 13,000
fresh_s1 = {}
country_counts = Counter()

with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
    f.readline()
    for idx, line in enumerate(f):
        if idx < 10000:
            continue
        if len(fresh_s1) >= 3000:
            break
        parts = line.rstrip("\r\n").split("\t")
        if len(parts) < 4: continue
        sid = parts[0].strip()
        c = normalize_country(parts[3])
        n = normalize_business_name(parts[1])
        a = normalize_business_address(parts[2])
        fresh_s1[sid] = (sid, n, a, c)
        country_counts[c] += 1

fresh_ids = set(fresh_s1.keys())

# 1. Verification of Independence
overlap = len(fresh_ids & orig_5k_ids)
assert overlap == 0, f"Error: {overlap} overlapping IDs!"

# Compute SHA-256 of sorted fresh IDs
id_string = "\n".join(sorted(fresh_ids))
sample_id_checksum = hashlib.sha256(id_string.encode("utf-8")).hexdigest()

print("=" * 65)
print("STRICT FRESH VALIDATION SAMPLE INTEGRITY")
print("=" * 65)
print(f"Sample Size:            {len(fresh_s1):,} S1 entities")
print(f"Overlap with Original:  {overlap} (0.00% overlap - strictly disjoint)")
print(f"Sample ID SHA-256:      {sample_id_checksum}")
print(f"Country Distribution:   {dict(country_counts)}")

# Load Ground Truth
fresh_gt = {}
with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        parts = line.rstrip("\r\n").split("\t")
        sid = parts[0].strip()
        if sid in fresh_ids:
            m_val = parts[1] if len(parts) > 1 else ""
            m_ids = {m.strip() for m in m_val.split(",") if m.strip()} if m_val else set()
            fresh_gt[sid] = m_ids
            if len(fresh_gt) == len(fresh_ids):
                break

total_true_matches = sum(len(s) for s in fresh_gt.values())
singletons = sum(1 for s in fresh_gt.values() if len(s) == 0)
print(f"Ground Truth Matches:   {total_true_matches:,} pairs")
print(f"Ground Truth Zero-Match:{singletons:,} ({singletons/len(fresh_gt)*100:.2f}%)")

# Collect target keys for fresh S1
target_keys = {"US": set(), "INDIA": set()}
for sid, (_, n, a, c) in fresh_s1.items():
    target_keys[c].update(get_v2_blocking_keys(n))

print(f"Target Keys:            US={len(target_keys['US']):,}, INDIA={len(target_keys['INDIA']):,}")

# Stream candidates from S2 and S3 for these fresh target keys
indexes = {"US": defaultdict(list), "INDIA": defaultdict(list)}
cand_store = {}
MAX_BUCKET = 120

for s_path, label in [("dataset/train/train_source2.tsv", "S2"), ("dataset/train/train_source3.tsv", "S3")]:
    t0 = time.time()
    with open(s_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4: continue
            c = normalize_country(parts[3])
            tkeys = target_keys.get(c)
            if not tkeys: continue
            
            cname = parts[1]
            norm_n = normalize_business_name(cname)
            keys = get_v2_blocking_keys(norm_n)
            matching = [k for k in keys if k in tkeys]
            if matching:
                cid = parts[0].strip()
                norm_a = normalize_business_address(parts[2])
                cand_store[cid] = (norm_n, norm_a)
                for mk in matching:
                    indexes[c][mk].append(cid)
    print(f"{label} streamed in {time.time() - t0:.2f}s. Stored candidates: {len(cand_store):,}")

for c in indexes:
    oversized = [k for k, v in indexes[c].items() if len(v) > MAX_BUCKET]
    for k in oversized:
        del indexes[c][k]

# Evaluate Candidate Recall
retrieved_true = 0
cands_by_s1 = {}
for sid, (_, s1_name, _, country) in fresh_s1.items():
    keys = get_v2_blocking_keys(s1_name)
    c_ids = set()
    idx = indexes.get(country, {})
    for k in keys:
        if k in idx:
            c_ids.update(idx[k])
    cands_by_s1[sid] = c_ids
    retrieved_true += len(c_ids & fresh_gt[sid])

cand_rec = retrieved_true / total_true_matches * 100 if total_true_matches else 0.0
print(f"Candidate Retrieval Recall: {retrieved_true:,} / {total_true_matches:,} ({cand_rec:.2f}%)")

# Compare V1 vs V3 on exact same candidates
evaluations = [
    ("Baseline V1 Matcher @ 0.75", "v1", 0.75),
    ("Corrected V3 Matcher @ 0.75 (Production)", "v3", SELECTED_THRESHOLD),
]

print("\n" + "=" * 65)
print("BENCHMARK ON FRESH VALIDATION SAMPLE")
print("=" * 65)

for label, m_type, th in evaluations:
    preds = {}
    total_p = 0
    tp_total = 0
    for sid, (_, s1_name, s1_addr, country) in fresh_s1.items():
        matched = []
        for cid in sorted(cands_by_s1[sid]):
            if cid not in cand_store: continue
            cn, ca = cand_store[cid]
            if m_type == "v1":
                feats = compute_pair_features(s1_name, s1_addr, cn, ca)
                score = score_candidate_pair(feats)
            else:
                score = score_pair_v3(s1_name, s1_addr, cn, ca)
            if score >= th:
                matched.append(cid)
        preds[sid] = set(matched)
        total_p += len(matched)
        tp_total += len(set(matched) & fresh_gt[sid])

    f05 = compute_f05_score(fresh_gt, preds)
    p = tp_total / total_p if total_p > 0 else 0.0
    r = tp_total / total_true_matches if total_true_matches > 0 else 0.0
    zero_rows = sum(1 for pset in preds.values() if len(pset) == 0)

    print(f"\n{label}:")
    print(f"  Macro F0.5:         {f05:.4f}")
    print(f"  Micro Precision:    {p:.4f} ({tp_total:,} / {total_p:,})")
    print(f"  Micro Recall:       {r:.4f} ({tp_total:,} / {total_true_matches:,})")
    print(f"  Predicted Matches:  {total_p:,} (TP: {tp_total:,}, FP: {total_p - tp_total:,})")
    print(f"  Zero-Match Rows:    {zero_rows:,} / {len(fresh_s1):,} ({zero_rows/len(fresh_s1)*100:.2f}%)")
