"""Diagnostic script to compare V1 and V2 configurations on identical validation entities.

Runs:
A. V1 Unchanged: V1 blocking candidates + V1 matcher @ 0.75
B. V2 Candidates + V1 Matcher @ 0.75
C. V2 Candidates + V2 Matcher @ 0.75
D. V2 Candidates + V2 Matcher @ 0.72

Measures:
- Macro F0.5 (official formula)
- Micro Precision and Recall
- Candidate Recall
- Predicted Match Count
- Zero-match entities
- Precision of matches added by V2 vs V1
- Correctness of matches removed by V2 vs V1
"""

import sys
import pickle
from pathlib import Path
from collections import defaultdict

src_dir = Path(__file__).resolve().parent / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(src_dir))

from features import compute_pair_features, score_candidate_pair
from predict_v2 import fast_score_candidate_pair, get_v2_blocking_keys
from blocking import get_blocking_keys
from evaluation import compute_f05_score

# Load validation caches
with open("output/val_cache_5k.pkl", "rb") as f:
    s1_records_v1, gt_records, old_cands_by_s1 = pickle.load(f)

with open("output/val_cache_v2.pkl", "rb") as f:
    s1_records_v2, _, indexes_v2, cand_store_v2 = pickle.load(f)

print(f"Validation entities: {len(gt_records):,}")
total_true_matches = sum(len(m) for m in gt_records.values())
print(f"Total true matches in ground truth: {total_true_matches:,}")

# Build V1 candidate mapping: s1_id -> set of candidate IDs
v1_cand_map = {}
v1_cand_records = {} # cid -> (norm_name, norm_addr)
for s1_id, recs in old_cands_by_s1.items():
    v1_cand_map[s1_id] = {cid for cid, _, _ in recs}
    for cid, cname, caddr in recs:
        v1_cand_records[cid] = (cname, caddr)

# Build V2 candidate mapping: s1_id -> set of candidate IDs
v2_cand_map = {}
for s1_id, (_, s1_name, _, country) in s1_records_v2.items():
    keys = get_v2_blocking_keys(s1_name)
    c_ids = set()
    idx = indexes_v2.get(country, {})
    for k in keys:
        if k in idx:
            c_ids.update(idx[k])
    v2_cand_map[s1_id] = c_ids

# Candidate Recalls
v1_cand_retrieved = sum(len(v1_cand_map[s1_id] & gt_records[s1_id]) for s1_id in gt_records)
v2_cand_retrieved = sum(len(v2_cand_map[s1_id] & gt_records[s1_id]) for s1_id in gt_records)
print(f"V1 Candidate Recall: {v1_cand_retrieved:,} / {total_true_matches:,} ({v1_cand_retrieved/total_true_matches*100:.2f}%)")
print(f"V2 Candidate Recall: {v2_cand_retrieved:,} / {total_true_matches:,} ({v2_cand_retrieved/total_true_matches*100:.2f}%)")

# Configurations to run
configs = [
    ("A. V1 Unchanged (V1 Cands + V1 Matcher @ 0.75)", "v1_cand", "v1_match", 0.75),
    ("B. Expanded V2 Cands + V1 Matcher @ 0.75",     "v2_cand", "v1_match", 0.75),
    ("C. V2 Matcher @ 0.75 (V2 Cands + V2 Matcher)", "v2_cand", "v2_match", 0.75),
    ("D. V2 Matcher @ 0.72 (V2 Cands + V2 Matcher)", "v2_cand", "v2_match", 0.72),
]

results = {}

for label, cand_src, match_src, threshold in configs:
    preds = {}
    total_preds = 0
    tp_total = 0
    fp_total = 0
    
    for s1_id, (_, s1_name, s1_addr, country) in s1_records_v2.items():
        if cand_src == "v1_cand":
            cand_ids = v1_cand_map.get(s1_id, set())
            store = v1_cand_records
        else:
            cand_ids = v2_cand_map.get(s1_id, set())
            store = cand_store_v2
            
        matched = []
        for cid in sorted(cand_ids):
            if cid not in store: continue
            cname, caddr = store[cid]
            if match_src == "v1_match":
                feats = compute_pair_features(s1_name, s1_addr, cname, caddr)
                score = score_candidate_pair(feats)
            else:
                score = fast_score_candidate_pair(s1_name, s1_addr, cname, caddr)
                
            if score >= threshold:
                matched.append(cid)
                
        preds[s1_id] = set(matched)
        total_preds += len(matched)
        true_set = gt_records.get(s1_id, set())
        tp = len(set(matched) & true_set)
        tp_total += tp
        fp_total += (len(matched) - tp)
        
    f05 = compute_f05_score(gt_records, preds)
    p = tp_total / total_preds if total_preds > 0 else 0.0
    r = tp_total / total_true_matches if total_true_matches > 0 else 0.0
    zero_matches = sum(1 for pset in preds.values() if len(pset) == 0)
    
    results[label] = {
        "f05": f05, "p": p, "r": r, "preds": total_preds,
        "tp": tp_total, "fp": fp_total, "zero_matches": zero_matches,
        "pred_dict": preds
    }
    
    print("\n" + "=" * 65)
    print(f"{label}")
    print("=" * 65)
    print(f"  Macro F0.5:         {f05:.4f}")
    print(f"  Micro Precision:    {p:.4f} ({tp_total:,} / {total_preds:,})")
    print(f"  Micro Recall:       {r:.4f} ({tp_total:,} / {total_true_matches:,})")
    print(f"  Predicted Matches:  {total_preds:,} (TP: {tp_total:,}, FP: {fp_total:,})")
    print(f"  Zero-Match Rows:    {zero_matches:,} / {len(s1_records_v2):,} ({zero_matches/len(s1_records_v2)*100:.2f}%)")

# Compare matches added and removed between A (V1) and D (V2 @ 0.72)
preds_a = results["A. V1 Unchanged (V1 Cands + V1 Matcher @ 0.75)"]["pred_dict"]
preds_d = results["D. V2 Matcher @ 0.72 (V2 Cands + V2 Matcher)"]["pred_dict"]

added_matches = 0
added_tp = 0
added_fp = 0
removed_matches = 0
removed_tp = 0
removed_fp = 0

for s1_id in gt_records:
    ma = preds_a.get(s1_id, set())
    md = preds_d.get(s1_id, set())
    true_set = gt_records[s1_id]
    
    # Matches in D but not A
    for cid in (md - ma):
        added_matches += 1
        if cid in true_set:
            added_tp += 1
        else:
            added_fp += 1
            
    # Matches in A but not D
    for cid in (ma - md):
        removed_matches += 1
        if cid in true_set:
            removed_tp += 1
        else:
            removed_fp += 1

print("\n" + "=" * 65)
print("MATCH DELTA ANALYSIS (V2 @ 0.72 vs V1 @ 0.75)")
print("=" * 65)
print(f"Matches Added by V2:   {added_matches:,}")
print(f"  - True Positives:    {added_tp:,}")
print(f"  - False Positives:   {added_fp:,}")
print(f"  - Precision of Added: {added_tp/added_matches*100:.2f}%" if added_matches else "  N/A")
print(f"Matches Removed by V2: {removed_matches:,}")
print(f"  - True Positives:    {removed_tp:,} (matches incorrectly dropped)")
print(f"  - False Positives:   {removed_fp:,} (false positives correctly dropped)")
