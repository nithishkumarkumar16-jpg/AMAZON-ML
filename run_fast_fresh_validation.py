"""Fast, Genuine Fresh Validation Suite for 1,000 S1 Entities.

Sample: Lines 20,000 to 25,000 of train_source1.tsv.
Strict 0% overlap with original 5k dev sample.
Evaluates:
  A. Baseline V1 Matcher @ 0.75
  B. Flawed Production V2 Matcher @ 0.72
  C. Corrected Production V3 Matcher @ 0.75 (score_pair_v3 imported directly from predict_v3)
"""

import hashlib
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

src_dir = Path(__file__).resolve().parent / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(src_dir))

# DIRECT IMPORT OF SHARED SCORING FUNCTION FROM PRODUCTION MODULE
from predict_v3 import score_pair_v3, SELECTED_THRESHOLD
from predict_v2 import fast_score_candidate_pair, get_v2_blocking_keys
from features import compute_pair_features, score_candidate_pair
from evaluation import compute_f05_score
from normalization import normalize_business_name, normalize_business_address, normalize_country

# 1. Verify 0% overlap with original 5k
with open("output/val_cache_5k.pkl", "rb") as f:
    orig_s1, _, _ = pickle.load(f)
orig_5k_ids = set(orig_s1.keys())

fresh_s1 = {}
country_counts = Counter()

with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
    f.readline()
    for idx, line in enumerate(f):
        if idx < 20000: continue
        if len(fresh_s1) >= 1000: break
        parts = line.rstrip("\r\n").split("\t")
        if len(parts) < 4: continue
        sid = parts[0].strip()
        c = normalize_country(parts[3])
        n = normalize_business_name(parts[1])
        a = normalize_business_address(parts[2])
        fresh_s1[sid] = (sid, n, a, c)
        country_counts[c] += 1

fresh_ids = set(fresh_s1.keys())
overlap = len(fresh_ids & orig_5k_ids)
assert overlap == 0, f"Error: {overlap} overlap!"

sample_id_checksum = hashlib.sha256("\n".join(sorted(fresh_ids)).encode("utf-8")).hexdigest()

print("=" * 65, flush=True)
print("GENUINE FRESH VALIDATION SAMPLE VERIFICATION", flush=True)
print("=" * 65, flush=True)
print(f"Sample Size:            {len(fresh_s1):,} S1 entities", flush=True)
print(f"Overlap with Dev 5k:    {overlap} (0.00% overlap - strictly disjoint)", flush=True)
print(f"Sample ID SHA-256:      {sample_id_checksum}", flush=True)
print(f"Country Distribution:   {dict(country_counts)}", flush=True)

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
            if len(fresh_gt) == len(fresh_ids): break

total_gt = sum(len(s) for s in fresh_gt.values())
singletons = sum(1 for s in fresh_gt.values() if len(s) == 0)
print(f"Ground Truth Matches:   {total_gt:,} true match pairs", flush=True)
print(f"Ground Truth Zero-Match:{singletons:,} ({singletons/len(fresh_gt)*100:.2f}%)", flush=True)

# Collect Target Keys
target_keys = {"US": set(), "INDIA": set()}
for sid, (_, n, a, c) in fresh_s1.items():
    target_keys[c].update(get_v2_blocking_keys(n))

print(f"Target Keys Generated:  US={len(target_keys['US']):,}, INDIA={len(target_keys['INDIA']):,}", flush=True)

# Stream Candidates from S2 and S3
indexes = {"US": defaultdict(list), "INDIA": defaultdict(list)}
cand_store = {}
MAX_BUCKET = 120

for s_path, label in [("dataset/train/train_source2.tsv", "S2"), ("dataset/train/train_source3.tsv", "S3")]:
    t0 = time.time()
    print(f"Streaming {label}...", flush=True)
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
    print(f"  {label} done in {time.time() - t0:.2f}s. Stored candidates so far: {len(cand_store):,}", flush=True)

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

cand_rec = retrieved_true / total_gt * 100 if total_gt else 0.0
print(f"\nCandidate Retrieval Recall: {retrieved_true:,} / {total_gt:,} ({cand_rec:.2f}%)", flush=True)

# Evaluate the 3 matchers
evaluations = [
    ("A. Baseline V1 Matcher @ 0.75", "v1", 0.75),
    ("B. Flawed Production V2 Matcher @ 0.72", "v2", 0.72),
    ("C. Corrected Production V3 Matcher @ 0.75 (Shared score_pair_v3)", "v3", SELECTED_THRESHOLD),
]

print("\n" + "=" * 65, flush=True)
print("HEAD-TO-HEAD RESULTS ON FRESH VALIDATION SAMPLE", flush=True)
print("=" * 65, flush=True)

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
            elif m_type == "v2":
                score = fast_score_candidate_pair(s1_name, s1_addr, cn, ca)
            else:
                score = score_pair_v3(s1_name, s1_addr, cn, ca)
            if score >= th:
                matched.append(cid)
        preds[sid] = set(matched)
        total_p += len(matched)
        tp_total += len(set(matched) & fresh_gt[sid])

    f05 = compute_f05_score(fresh_gt, preds)
    p = tp_total / total_p if total_p > 0 else 0.0
    r = tp_total / total_gt if total_gt > 0 else 0.0
    zero_rows = sum(1 for pset in preds.values() if len(pset) == 0)

    print(f"\n{label}:", flush=True)
    print(f"  Macro F0.5:         {f05:.4f}", flush=True)
    print(f"  Micro Precision:    {p:.4f} ({tp_total:,} / {total_p:,})", flush=True)
    print(f"  Micro Recall:       {r:.4f} ({tp_total:,} / {total_gt:,})", flush=True)
    print(f"  Predicted Matches:  {total_p:,} (TP: {tp_total:,}, FP: {total_p - tp_total:,})", flush=True)
    print(f"  Zero-Match Rows:    {zero_rows:,} / {len(fresh_s1):,} ({zero_rows/len(fresh_s1)*100:.2f}%)", flush=True)
