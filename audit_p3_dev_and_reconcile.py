import hashlib
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
src_dir = Path("code/business_entity_resolution/src").resolve()
sys.path.insert(0, str(src_dir))

from evaluation import compute_f05_score

print("=" * 70)
print("AUDIT AND RECONCILIATION FROM SAVED EXPORTED FILES")
print("=" * 70)

# 1. Inspect and Hash Exported Files
files_to_check = [
    "output/dev_1k_v1_candidates.tsv",
    "output/dev_1k_p3_candidates.tsv",
    "output/dev_1k_v1_predictions.tsv",
    "output/dev_1k_p3_predictions.tsv"
]

print("\n--- 1. FILE CHECKSUMS AND SIZES ---")
for fpath in files_to_check:
    hasher = hashlib.sha256()
    size = os.path.getsize(fpath)
    with open(fpath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    sha = hasher.hexdigest()
    print(f"File: {fpath}")
    print(f"  Size:   {size:,} bytes")
    print(f"  SHA256: {sha}")

# 2. Load Ground Truth for Dev 1k
dev_s1_ids = []
with open("output/dev_1k_v1_predictions.tsv", "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        parts = line.rstrip("\r\n").split("\t")
        if parts:
            dev_s1_ids.append(parts[0].strip())

dev_s1_set = set(dev_s1_ids)

dev_gt = {}
with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        parts = line.rstrip("\r\n").split("\t")
        sid = parts[0].strip()
        if sid in dev_s1_set:
            m_val = parts[1] if len(parts) > 1 else ""
            m_ids = {m.strip() for m in m_val.split(",") if m.strip()} if m_val else set()
            dev_gt[sid] = m_ids
            if len(dev_gt) == len(dev_s1_set):
                break

total_gt_pairs = sum(len(s) for s in dev_gt.values())
zero_match_gt = sum(1 for s in dev_gt.values() if len(s) == 0)
print(f"\nGround Truth for Dev Set ({len(dev_s1_ids)} S1 entities):")
print(f"  Total True Matches:  {total_gt_pairs:,} pairs")
print(f"  Zero-Match Entities: {zero_match_gt} / {len(dev_s1_ids)} ({zero_match_gt/len(dev_s1_ids)*100:.2f}%)")

# 3. Load Candidates and Evaluate Candidate Recall
def load_candidates(fpath):
    cands_per_s1 = defaultdict(set)
    total_pairs = 0
    with open(fpath, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 2:
                sid, cid = parts[0].strip(), parts[1].strip()
                cands_per_s1[sid].add(cid)
                total_pairs += 1
    return cands_per_s1, total_pairs

v1_cands, v1_cand_count = load_candidates("output/dev_1k_v1_candidates.tsv")
p3_cands, p3_cand_count = load_candidates("output/dev_1k_p3_candidates.tsv")

def eval_cand_recall(cands_dict):
    retrieved_true = 0
    for sid, true_set in dev_gt.items():
        cand_set = cands_dict.get(sid, set())
        retrieved_true += len(cand_set & true_set)
    return retrieved_true, retrieved_true / total_gt_pairs * 100

v1_ret_true, v1_cand_rec = eval_cand_recall(v1_cands)
p3_ret_true, p3_cand_rec = eval_cand_recall(p3_cands)

print("\n--- 2. CANDIDATE RETRIEVAL METRICS (FROM SAVED FILES) ---")
print(f"V1 Candidates: {v1_cand_count:,} pairs | True Retrieved: {v1_ret_true:,} / {total_gt_pairs:,} ({v1_cand_rec:.2f}%)")
print(f"P3 Candidates: {p3_cand_count:,} pairs | True Retrieved: {p3_ret_true:,} / {total_gt_pairs:,} ({p3_cand_rec:.2f}%)")

# 4. Load Predictions and Recompute Metrics
def load_predictions(fpath):
    preds = {}
    with open(fpath, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if not parts: continue
            sid = parts[0].strip()
            m_val = parts[1] if len(parts) > 1 else ""
            m_ids = {m.strip() for m in m_val.split(",") if m.strip()} if m_val else set()
            preds[sid] = m_ids
    return preds

v1_preds = load_predictions("output/dev_1k_v1_predictions.tsv")
p3_preds = load_predictions("output/dev_1k_p3_predictions.tsv")

def compute_metrics(preds):
    macro_f05 = compute_f05_score(dev_gt, preds)
    tp, fp, fn = 0, 0, 0
    zm_count = 0
    pred_pairs = set()
    for sid in dev_s1_ids:
        p_set = set(preds.get(sid, []))
        g_set = dev_gt.get(sid, set())
        if not p_set:
            zm_count += 1
        for cid in p_set:
            pred_pairs.add((sid, cid))
        tp += len(p_set & g_set)
        fp += len(p_set - g_set)
        fn += len(g_set - p_set)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return macro_f05, prec, rec, tp, fp, fn, zm_count, pred_pairs

f05_v1, p_v1, r_v1, tp_v1, fp_v1, fn_v1, zm_v1, pairs_v1 = compute_metrics(v1_preds)
f05_p3, p_p3, r_p3, tp_p3, fp_p3, fn_p3, zm_p3, pairs_p3 = compute_metrics(p3_preds)

print("\n--- 3. RECOMPUTED AGGREGATE METRICS (FROM SAVED PREDICTIONS) ---")
print(f"{'Config':<10} | {'Macro F0.5':<10} | {'Precision':<9} | {'Recall':<8} | {'TP':<6} | {'FP':<6} | {'FN':<6} | {'Zero-Match':<10}")
print("-" * 78)
print(f"{'V1 Dev':<10} | {f05_v1:.4f}     | {p_v1*100:.2f}%   | {r_v1*100:.2f}% | {tp_v1:<6} | {fp_v1:<6} | {fn_v1:<6} | {zm_v1} / {len(dev_s1_ids)}")
print(f"{'P3 Dev':<10} | {f05_p3:.4f}     | {p_p3*100:.2f}%   | {r_p3*100:.2f}% | {tp_p3:<6} | {fp_p3:<6} | {fn_p3:<6} | {zm_p3} / {len(dev_s1_ids)}")
print("-" * 78)

# 5. Exact Set Difference Reconciliation (Added vs Removed)
# True match pairs in GT:
gt_pairs = {(sid, cid) for sid, cids in dev_gt.items() for cid in cids}

v1_tp_set = pairs_v1 & gt_pairs
v1_fp_set = pairs_v1 - gt_pairs

p3_tp_set = pairs_p3 & gt_pairs
p3_fp_set = pairs_p3 - gt_pairs

added_pairs = pairs_p3 - pairs_v1
removed_pairs = pairs_v1 - pairs_p3
common_pairs = pairs_v1 & pairs_p3

added_tp = added_pairs & gt_pairs
added_fp = added_pairs - gt_pairs

removed_tp = removed_pairs & gt_pairs
removed_fp = removed_pairs - gt_pairs

common_tp = common_pairs & gt_pairs
common_fp = common_pairs - gt_pairs

print("\n--- 4. EXACT SET DIFFERENCE RECONCILIATION ---")
print(f"Total V1 Predictions:      {len(pairs_v1):,} (TP: {len(v1_tp_set):,}, FP: {len(v1_fp_set):,})")
print(f"Total P3 Predictions:      {len(pairs_p3):,} (TP: {len(p3_tp_set):,}, FP: {len(p3_fp_set):,})")
print(f"Common Predictions:        {len(common_pairs):,} (TP: {len(common_tp):,}, FP: {len(common_fp):,})")
print(f"Added Predictions (P3 - V1): {len(added_pairs):,} (Added TP: {len(added_tp):,}, Added FP: {len(added_fp):,})")
print(f"Removed Predictions (V1 - P3): {len(removed_pairs):,} (Removed TP: {len(removed_tp):,}, Removed FP: {len(removed_fp):,})")

delta_tp = len(p3_tp_set) - len(v1_tp_set)
delta_fp = len(p3_fp_set) - len(v1_fp_set)
computed_delta_tp = len(added_tp) - len(removed_tp)
computed_delta_fp = len(added_fp) - len(removed_fp)

print("\nReconciliation Verification:")
print(f"  Δ TP: {len(p3_tp_set)} - {len(v1_tp_set)} = {delta_tp:+d}")
print(f"  Computed from set diffs: {len(added_tp)} (Added TP) - {len(removed_tp)} (Removed TP) = {computed_delta_tp:+d}")
print(f"  Check: {delta_tp == computed_delta_tp}")

print(f"  Δ FP: {len(p3_fp_set)} - {len(v1_fp_set)} = {delta_fp:+d}")
print(f"  Computed from set diffs: {len(added_fp)} (Added FP) - {len(removed_fp)} (Removed FP) = {computed_delta_fp:+d}")
print(f"  Check: {delta_fp == computed_delta_fp}")

# Inspect reasons for Removed TP and Removed FP
print("\nSample of Removed False Positives (V1 accepted, P3 rejected):")
count = 0
for sid, cid in list(removed_fp)[:5]:
    count += 1
    print(f"  {count}. S1: {sid} -> Cand: {cid}")

print("\nSample of Removed True Positives (V1 accepted, P3 rejected):")
count = 0
for sid, cid in list(removed_tp)[:5]:
    count += 1
    print(f"  {count}. S1: {sid} -> Cand: {cid}")
