"""Deep diagnostic of Baseline V1 on 5,000 S1 validation split.

Performs:
1. Complete False Negative decomposition (Retrieved vs Never-Retrieved)
2. Oracle Candidate-Ceiling Score
3. Separate S2 and S3 recall and matching metrics
4. Country slices (US vs India)
5. Missing-field slices
6. Cardinality slices (zero, 1, multiple matches)
7. Micro and Macro metrics
"""

import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

src_dir = Path(__file__).resolve().parent / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(src_dir))

from features import compute_pair_features, score_candidate_pair
from matcher import BaselineMatcher
from evaluation import compute_f05_score

CACHE_FILE = Path("output/val_cache_5k.pkl")
if not CACHE_FILE.exists():
    print(f"Error: {CACHE_FILE} not found!")
    sys.exit(1)

with open(CACHE_FILE, "rb") as f:
    s1_records, gt_records, candidates_by_s1 = pickle.load(f)

print(f"Loaded validation cache: {len(s1_records):,} S1 entities.")

# --- 1. Candidate Retrieval / Blocking Diagnosis ---
total_true_matches = sum(len(m) for m in gt_records.values())
true_s2_matches = sum(len([x for x in m if x.startswith("S2-")]) for m in gt_records.values())
true_s3_matches = sum(len([x for x in m if x.startswith("S3-")]) for m in gt_records.values())

retrieved_true_matches = 0
retrieved_true_s2 = 0
retrieved_true_s3 = 0

cand_counts = []
cand_oracle_preds = {}
entities_all_retrieved = 0
entities_partial_retrieved = 0
entities_zero_retrieved = 0
matchable_entities = 0

for s1_id, (_, _, _, _) in s1_records.items():
    c_records = candidates_by_s1.get(s1_id, [])
    c_ids = {cid for cid, _, _ in c_records}
    cand_counts.append(len(c_ids))
    true_set = gt_records.get(s1_id, set())

    # Oracle prediction: predict all true matches that are in candidates
    oracle_set = true_set & c_ids
    cand_oracle_preds[s1_id] = oracle_set

    found = len(oracle_set)
    retrieved_true_matches += found
    retrieved_true_s2 += len([x for x in oracle_set if x.startswith("S2-")])
    retrieved_true_s3 += len([x for x in oracle_set if x.startswith("S3-")])

    if len(true_set) > 0:
        matchable_entities += 1
        if found == len(true_set):
            entities_all_retrieved += 1
        elif found > 0:
            entities_partial_retrieved += 1
        else:
            entities_zero_retrieved += 1

print("\n" + "=" * 70)
print("1. CANDIDATE RETRIEVAL (BLOCKING) DIAGNOSIS")
print("=" * 70)
print(f"Total True Matches: {total_true_matches:,} (S2: {true_s2_matches:,}, S3: {true_s3_matches:,})")
print(f"Retrieved in Candidates: {retrieved_true_matches:,} / {total_true_matches:,} ({retrieved_true_matches/total_true_matches*100:.2f}%)")
print(f"  - S2 Candidate Recall: {retrieved_true_s2:,} / {true_s2_matches:,} ({retrieved_true_s2/true_s2_matches*100:.2f}%)")
print(f"  - S3 Candidate Recall: {retrieved_true_s3:,} / {true_s3_matches:,} ({retrieved_true_s3/true_s3_matches*100:.2f}%)")
print(f"Missed in Candidate Generation (Hard FN): {total_true_matches - retrieved_true_matches:,} ({100 - retrieved_true_matches/total_true_matches*100:.2f}%)")

cand_counts_sorted = sorted(cand_counts)
n = len(cand_counts)
print(f"Candidate count per entity: Mean={sum(cand_counts)/n:.2f}, Median={cand_counts_sorted[n//2]}, P95={cand_counts_sorted[int(n*0.95)]}, P99={cand_counts_sorted[int(n*0.99)]}, Max={cand_counts_sorted[-1]}")
print(f"Matchable Entities Coverage: 100% Retrieved: {entities_all_retrieved:,} ({entities_all_retrieved/matchable_entities*100:.2f}%), Partial: {entities_partial_retrieved:,} ({entities_partial_retrieved/matchable_entities*100:.2f}%), None: {entities_zero_retrieved:,} ({entities_zero_retrieved/matchable_entities*100:.2f}%)")

# Oracle Score
oracle_f05 = compute_f05_score(gt_records, cand_oracle_preds)
print(f"\n>>> ORACLE CANDIDATE-CEILING Macro F0.5: {oracle_f05:.4f} <<<")
print("  (This is the theoretical maximum F0.5 achievable without improving candidate retrieval)")

# --- 2. Baseline V1 Matcher Evaluation (Threshold 0.75) ---
print("\n" + "=" * 70)
print("2. BASELINE V1 MATCHER EVALUATION (Threshold = 0.75)")
print("=" * 70)

THRESHOLD = 0.75
v1_preds = {}
total_pred_matches = 0
tp_total = 0
fp_total = 0
fn_total = 0

s2_tp = 0
s2_fp = 0
s2_fn = 0
s3_tp = 0
s3_fp = 0
s3_fn = 0

fn_unretrieved = 0  # True match not in candidate set
fn_rejected = 0     # True match in candidate set but score < 0.75

# Country slices
country_metrics = defaultdict(lambda: {"gt": {}, "pred": {}, "tp": 0, "fp": 0, "fn": 0})
# Missing address slices
missing_addr_metrics = defaultdict(lambda: {"gt": {}, "pred": {}, "tp": 0, "fp": 0, "fn": 0})
# Cardinality slices
card_metrics = defaultdict(lambda: {"gt": {}, "pred": {}, "tp": 0, "fp": 0, "fn": 0})

for s1_id, (_, s1_name, s1_addr, country) in s1_records.items():
    cands = candidates_by_s1.get(s1_id, [])
    true_set = gt_records.get(s1_id, set())

    matched = []
    cand_ids_seen = set()
    for cid, cname, caddr in cands:
        cand_ids_seen.add(cid)
        feats = compute_pair_features(s1_name, s1_addr, cname, caddr)
        score = score_candidate_pair(feats)
        if score >= THRESHOLD:
            matched.append(cid)

    pred_set = set(matched)
    v1_preds[s1_id] = pred_set
    total_pred_matches += len(pred_set)

    # TP, FP, FN
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    tp_total += tp
    fp_total += fp
    fn_total += fn

    # Source slices
    for m in pred_set:
        if m.startswith("S2-"):
            if m in true_set: s2_tp += 1
            else: s2_fp += 1
        elif m.startswith("S3-"):
            if m in true_set: s3_tp += 1
            else: s3_fp += 1

    for m in true_set:
        if m not in pred_set:
            if m.startswith("S2-"): s2_fn += 1
            elif m.startswith("S3-"): s3_fn += 1

    # FN Decomposition
    for t_id in true_set:
        if t_id not in pred_set:
            if t_id not in cand_ids_seen:
                fn_unretrieved += 1
            else:
                fn_rejected += 1

    # Slices
    country_metrics[country]["gt"][s1_id] = true_set
    country_metrics[country]["pred"][s1_id] = pred_set
    country_metrics[country]["tp"] += tp
    country_metrics[country]["fp"] += fp
    country_metrics[country]["fn"] += fn

    addr_state = "Missing S1 Addr" if not s1_addr else "Has S1 Addr"
    missing_addr_metrics[addr_state]["gt"][s1_id] = true_set
    missing_addr_metrics[addr_state]["pred"][s1_id] = pred_set
    missing_addr_metrics[addr_state]["tp"] += tp
    missing_addr_metrics[addr_state]["fp"] += fp
    missing_addr_metrics[addr_state]["fn"] += fn

    card_label = "0 (Singleton)" if len(true_set) == 0 else ("1 Match" if len(true_set) == 1 else ">1 Matches")
    card_metrics[card_label]["gt"][s1_id] = true_set
    card_metrics[card_label]["pred"][s1_id] = pred_set
    card_metrics[card_label]["tp"] += tp
    card_metrics[card_label]["fp"] += fp
    card_metrics[card_label]["fn"] += fn

macro_f05 = compute_f05_score(gt_records, v1_preds)
micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
micro_f05 = (1.25 * micro_p * micro_r) / (0.25 * micro_p + micro_r) if (0.25 * micro_p + micro_r) > 0 else 0.0

print(f"Overall Macro F0.5: {macro_f05:.4f}")
print(f"Micro Metrics: P={micro_p:.4f}, R={micro_r:.4f}, F0.5={micro_f05:.4f}")
print(f"Totals: TP={tp_total:,}, FP={fp_total:,}, FN={fn_total:,}")
print(f"Matches: True={total_true_matches:,}, Predicted={total_pred_matches:,}")

print("\n--- FALSE NEGATIVE BOTTLENECK DECOMPOSITION ---")
print(f"Total False Negatives: {fn_total:,} (100.0%)")
print(f"  A. True matches NEVER RETRIEVED (Blocking failure): {fn_unretrieved:,} ({fn_unretrieved/fn_total*100:.2f}%)")
print(f"  B. True matches REJECTED by Matcher (Scoring failure): {fn_rejected:,} ({fn_rejected/fn_total*100:.2f}%)")
print(f">>> CONCLUSION: Blocking is responsible for {fn_unretrieved/fn_total*100:.1f}% of ALL False Negatives! <<<")

print("\n--- SEPARATE SOURCE BREAKDOWN ---")
s2_p = s2_tp / (s2_tp + s2_fp) if (s2_tp + s2_fp) > 0 else 0
s2_r = s2_tp / (s2_tp + s2_fn) if (s2_tp + s2_fn) > 0 else 0
s3_p = s3_tp / (s3_tp + s3_fp) if (s3_tp + s3_fp) > 0 else 0
s3_r = s3_tp / (s3_tp + s3_fn) if (s3_tp + s3_fn) > 0 else 0
print(f"Source 2: TP={s2_tp:,}, FP={s2_fp:,}, FN={s2_fn:,} | P={s2_p:.4f}, R={s2_r:.4f}")
print(f"Source 3: TP={s3_tp:,}, FP={s3_fp:,}, FN={s3_fn:,} | P={s3_p:.4f}, R={s3_r:.4f}")

print("\n--- COUNTRY SLICES ---")
for c, d in country_metrics.items():
    f05 = compute_f05_score(d["gt"], d["pred"])
    p = d["tp"] / (d["tp"] + d["fp"]) if (d["tp"] + d["fp"]) > 0 else 0
    r = d["tp"] / (d["tp"] + d["fn"]) if (d["tp"] + d["fn"]) > 0 else 0
    print(f"{c:<10}: Macro F0.5={f05:.4f} | Micro P={p:.4f}, R={r:.4f} (TP={d['tp']:,}, FP={d['fp']:,}, FN={d['fn']:,})")

print("\n--- CARDINALITY SLICES ---")
for card, d in card_metrics.items():
    f05 = compute_f05_score(d["gt"], d["pred"])
    p = d["tp"] / (d["tp"] + d["fp"]) if (d["tp"] + d["fp"]) > 0 else 0
    r = d["tp"] / (d["tp"] + d["fn"]) if (d["tp"] + d["fn"]) > 0 else 0
    print(f"{card:<15}: Macro F0.5={f05:.4f} | Micro P={p:.4f}, R={r:.4f} (Count={len(d['gt']):,})")

print("\n--- ADDRESS MISSINGNESS SLICES ---")
for addr_st, d in missing_addr_metrics.items():
    f05 = compute_f05_score(d["gt"], d["pred"])
    p = d["tp"] / (d["tp"] + d["fp"]) if (d["tp"] + d["fp"]) > 0 else 0
    r = d["tp"] / (d["tp"] + d["fn"]) if (d["tp"] + d["fn"]) > 0 else 0
    print(f"{addr_st:<20}: Macro F0.5={f05:.4f} | Micro P={p:.4f}, R={r:.4f}")
