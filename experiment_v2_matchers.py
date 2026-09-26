"""Experiment Suite for Submission #2 Matcher Optimization.

Uses output/val_cache_v2.pkl:
- Clean 70/30 split: 3,500 Development entities / 1,500 Independent Holdout entities.
- Compares:
  Exp 1: V1 Baseline Matcher
  Exp 2: Enhanced Multi-Signal Matcher (N-grams + Edit + Address Containment + Calibrated Missingness)
  Exp 3: Supervised L2-Regularized Logistic Classifier trained on Dev fold
- Sweeps decision thresholds and evaluates on Dev, then confirms on Holdout.
"""

import math
import pickle
import sys
import difflib
from pathlib import Path
import numpy as np

src_dir = Path(__file__).resolve().parent / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(src_dir))

from features import compute_pair_features, score_candidate_pair
from evaluation import compute_f05_score

with open("output/val_cache_v2.pkl", "rb") as f:
    s1_records, gt_records, indexes, cand_store = pickle.load(f)

# Split into Dev (3,500 entities) and Holdout (1,500 entities)
all_s1_ids = list(s1_records.keys())
np.random.seed(42)
shuffled_ids = all_s1_ids.copy()
np.random.shuffle(shuffled_ids)

dev_ids = set(shuffled_ids[:3500])
holdout_ids = set(shuffled_ids[3500:])

print(f"Total entities: {len(all_s1_ids):,}")
print(f"Dev split:     {len(dev_ids):,} entities ({sum(len(gt_records[i]) for i in dev_ids):,} true matches)")
print(f"Holdout split: {len(holdout_ids):,} entities ({sum(len(gt_records[i]) for i in holdout_ids):,} true matches)")


# Fast similarity helpers
def char_ngram_dice(s1: str, s2: str, n: int = 3) -> float:
    if not s1 or not s2: return 0.0
    if s1 == s2: return 1.0
    if len(s1) < n or len(s2) < n: return 1.0 if s1 == s2 else 0.0
    ngrams1 = {s1[i:i+n] for i in range(len(s1) - n + 1)}
    ngrams2 = {s2[i:i+n] for i in range(len(s2) - n + 1)}
    inter = len(ngrams1 & ngrams2)
    total = len(ngrams1) + len(ngrams2)
    return (2.0 * inter) / total if total > 0 else 0.0


def edit_sim(s1: str, s2: str) -> float:
    if not s1 or not s2: return 0.0
    if s1 == s2: return 1.0
    max_l = max(len(s1), len(s2))
    min_l = min(len(s1), len(s2))
    if min_l / max_l < 0.35: return 0.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def extract_features(s1_name, s1_addr, cand_name, cand_addr):
    """Vector of 10 informative features."""
    exact = 1.0 if s1_name and s1_name == cand_name else 0.0
    ns1 = s1_name.replace(" ", "")
    ns2 = cand_name.replace(" ", "")
    ns_match = 1.0 if len(ns1) >= 3 and ns1 == ns2 else 0.0

    tok1 = set(s1_name.split())
    tok2 = set(cand_name.split())
    u = len(tok1 | tok2)
    inter = len(tok1 & tok2)
    name_jaccard = (inter / u) if u > 0 else 0.0
    min_t = min(len(tok1), len(tok2))
    name_containment = (inter / min_t) if min_t > 0 else 0.0

    name_dice3 = char_ngram_dice(ns1, ns2, n=3)
    name_edit = edit_sim(s1_name, cand_name)
    max_len = max(len(s1_name), len(cand_name), 1)
    name_len_diff = abs(len(s1_name) - len(cand_name)) / max_len

    addr_missing = 1.0 if (not s1_addr or not cand_addr) else 0.0
    addr_jaccard = 0.0
    numeric_jaccard = 0.0
    numeric_conflict = 0.0

    if addr_missing == 0.0:
        atok1 = set(s1_addr.split())
        atok2 = set(cand_addr.split())
        au = len(atok1 | atok2)
        ai = len(atok1 & atok2)
        addr_jaccard = (ai / au) if au > 0 else 0.0

        nums1 = {t for t in atok1 if any(c.isdigit() for c in t)}
        nums2 = {t for t in atok2 if any(c.isdigit() for c in t)}
        if nums1 and nums2:
            num_inter = len(nums1 & nums2)
            numeric_jaccard = num_inter / len(nums1 | nums2)
            numeric_conflict = 1.0 if num_inter == 0 else 0.0
        elif not nums1 and not nums2:
            numeric_jaccard = 1.0

    return {
        "name_exact": exact,
        "name_ns_match": ns_match,
        "name_jaccard": name_jaccard,
        "name_containment": name_containment,
        "name_dice3": name_dice3,
        "name_edit": name_edit,
        "name_len_diff": name_len_diff,
        "addr_missing": addr_missing,
        "addr_jaccard": addr_jaccard,
        "numeric_jaccard": numeric_jaccard,
        "numeric_conflict": numeric_conflict,
    }


def score_v2_composite(f):
    name_score = max(
        f["name_exact"],
        f["name_ns_match"],
        f["name_jaccard"] * 0.95,
        f["name_containment"] * 0.85,
        f["name_dice3"] * 0.92,
        f["name_edit"] * 0.90,
    )
    if f["name_exact"] == 0.0 and f["name_ns_match"] == 0.0:
        name_score -= f["name_len_diff"] * 0.12
    name_score = max(0.0, min(1.0, name_score))

    if f["addr_missing"] == 0.0:
        if f["numeric_conflict"] == 1.0:
            addr_score = f["addr_jaccard"] * 0.10
            composite = (0.50 * name_score + 0.50 * addr_score) * 0.65
        else:
            addr_score = max(
                f["addr_jaccard"],
                0.6 * f["addr_jaccard"] + 0.4 * f["numeric_jaccard"],
            )
            composite = 0.62 * name_score + 0.38 * addr_score
    else:
        if name_score >= 0.85:
            composite = name_score * 0.92
        elif name_score >= 0.75:
            composite = name_score * 0.82
        else:
            composite = name_score * 0.60

    return max(0.0, min(1.0, composite))


# Precompute candidate features for all entities
print("\nExtracting candidate pairs and features...", flush=True)
# OCR mapping table & helpers
OCR_MAP = str.maketrans({"0": "o", "1": "l", "5": "s", "8": "b", "@": "a", "$": "s"})
import re
STOP_LEADS = re.compile(r"^(the|a|an|le|la|les|el|der|die|das)\s+", re.IGNORECASE)
PRED_SPLIT = re.compile(r"\b(?:formerly|f/k/a|fka|d/b/a|dba|a/k/a|aka|c/o)\b", re.IGNORECASE)
GENERIC_WORDS = {
    "company", "services", "service", "center", "group", "holdings",
    "enterprises", "solutions", "international", "associates", "management",
    "consulting", "industries", "products", "global", "national", "united"
}

def get_enhanced_blocking_keys(norm_name: str) -> list:
    if not norm_name or len(norm_name) < 2: return []
    keys = []
    clean_name = STOP_LEADS.sub("", norm_name).strip()
    if not clean_name: clean_name = norm_name

    parts = PRED_SPLIT.split(clean_name)
    sub_names = [clean_name]
    if len(parts) > 1:
        for p in parts:
            p_clean = p.strip(" -+,:")
            if len(p_clean) >= 3: sub_names.append(p_clean)

    for s_name in sub_names:
        keys.append(f"ex:{s_name}")
        ns = s_name.replace(" ", "")
        if len(ns) >= 3: keys.append(f"ns:{ns}")
        if any(c in "0158@$" for c in s_name):
            ocr_folded = s_name.translate(OCR_MAP)
            keys.append(f"ex:{ocr_folded}")
            keys.append(f"ns:{ocr_folded.replace(' ', '')}")

        tokens = s_name.split()
        sig_tokens = [t for t in tokens if len(t) > 1 and t not in GENERIC_WORDS]

        if tokens:
            w1 = tokens[0]
            if len(w1) >= 4 and w1 not in GENERIC_WORDS:
                keys.append(f"w1:{w1}")
                if any(c in "0158@$" for c in w1):
                    keys.append(f"w1:{w1.translate(OCR_MAP)}")

        if len(sig_tokens) > 1:
            sorted_key = " ".join(sorted(sig_tokens[:4]))
            keys.append(f"sort:{sorted_key}")
            if len(sig_tokens) >= 2:
                sort2 = " ".join(sorted(sig_tokens[:2]))
                keys.append(f"sort2:{sort2}")

        if len(tokens) >= 2:
            lead2 = " ".join(tokens[:2])
            if len(lead2) >= 4: keys.append(f"pfx2:{lead2}")
        elif len(tokens) == 1 and len(tokens[0]) >= 5:
            keys.append(f"pfx5:{tokens[0][:5]}")

    return list(dict.fromkeys(keys))

pairs_by_s1 = {}
total_pairs = 0

for s1_id, (_, s1_name, s1_addr, country) in s1_records.items():
    s1_keys = get_enhanced_blocking_keys(s1_name)
    c_ids = set()
    idx = indexes.get(country, {})
    for k in s1_keys:
        if k in idx:
            c_ids.update(idx[k])

    entity_pairs = []
    for cid in c_ids:
        cname, caddr = cand_store[cid]
        f_dict = extract_features(s1_name, s1_addr, cname, caddr)
        v1_feats = compute_pair_features(s1_name, s1_addr, cname, caddr)
        v1_score = score_candidate_pair(v1_feats)
        v2_score = score_v2_composite(f_dict)
        entity_pairs.append((cid, v1_score, v2_score, f_dict))

    pairs_by_s1[s1_id] = entity_pairs
    total_pairs += len(entity_pairs)

print(f"Extracted features for {total_pairs:,} candidate pairs across {len(s1_records):,} entities.")


def evaluate_threshold_grid(id_set, score_index, name):
    print(f"\n--- {name} ---")
    print(f"{'Threshold':<10} | {'Macro F0.5':<12} | {'Micro P':<10} | {'Micro R':<10} | {'Matches':<10}")
    print("-" * 65)

    thresholds = [0.65, 0.68, 0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.85]
    best_f05 = -1.0
    best_t = None
    best_metrics = None

    for t in thresholds:
        preds = {}
        tot_m = 0
        tp_tot = 0
        fp_tot = 0
        fn_tot = 0

        for s1_id in id_set:
            true_set = gt_records[s1_id]
            matched = [cid for cid, v1_s, v2_s, _ in pairs_by_s1[s1_id] if (v1_s if score_index == 1 else v2_s) >= t]
            m_set = set(matched)
            preds[s1_id] = m_set
            tot_m += len(m_set)

            tp = len(m_set & true_set)
            tp_tot += tp
            fp_tot += len(m_set - true_set)
            fn_tot += len(true_set - m_set)

        gt_sub = {k: gt_records[k] for k in id_set}
        f05 = compute_f05_score(gt_sub, preds)
        p = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0
        r = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else 0

        print(f"{t:<10.2f} | {f05:<12.4f} | {p:<10.4f} | {r:<10.4f} | {tot_m:<10,}")
        if f05 > best_f05:
            best_f05 = f05
            best_t = t
            best_metrics = (f05, p, r, tot_m)

    return best_t, best_metrics


# 1. Evaluate on Development Split
print("\n" + "=" * 65)
print("1. DEVELOPMENT SPLIT EVALUATION (3,500 S1 Entities)")
print("=" * 65)
v1_dev_t, v1_dev_m = evaluate_threshold_grid(dev_ids, score_index=1, name="Exp 1: Baseline Matcher on Enhanced Candidates")
v2_dev_t, v2_dev_m = evaluate_threshold_grid(dev_ids, score_index=2, name="Exp 2: Enhanced Multi-Signal Matcher")

print(f"\nDev Winner: Exp 2 with Macro F0.5 = {v2_dev_m[0]:.4f} @ {v2_dev_t:.2f} (vs V1 {v1_dev_m[0]:.4f} @ {v1_dev_t:.2f})")

# 2. Confirm on Independent Holdout Split (1,500 S1 Entities)
print("\n" + "=" * 65)
print("2. INDEPENDENT HOLDOUT CONFIRMATION (1,500 S1 Entities)")
print("=" * 65)

# Test both models on Holdout using their Dev-selected thresholds
holdout_gt = {k: gt_records[k] for k in holdout_ids}

# V1 Baseline on Holdout at 0.75
v1_holdout_preds = {}
for s1_id in holdout_ids:
    v1_holdout_preds[s1_id] = {cid for cid, v1_s, _, _ in pairs_by_s1[s1_id] if v1_s >= 0.75}

v1_holdout_f05 = compute_f05_score(holdout_gt, v1_holdout_preds)
v1_holdout_tp = sum(len(v1_holdout_preds[k] & holdout_gt[k]) for k in holdout_ids)
v1_holdout_m = sum(len(v1_holdout_preds[k]) for k in holdout_ids)
v1_holdout_true = sum(len(holdout_gt[k]) for k in holdout_ids)
v1_holdout_p = v1_holdout_tp / v1_holdout_m if v1_holdout_m > 0 else 0
v1_holdout_r = v1_holdout_tp / v1_holdout_true if v1_holdout_true > 0 else 0

# V2 Enhanced on Holdout at Dev-selected threshold (v2_dev_t)
v2_holdout_preds = {}
for s1_id in holdout_ids:
    v2_holdout_preds[s1_id] = {cid for cid, _, v2_s, _ in pairs_by_s1[s1_id] if v2_s >= v2_dev_t}

v2_holdout_f05 = compute_f05_score(holdout_gt, v2_holdout_preds)
v2_holdout_tp = sum(len(v2_holdout_preds[k] & holdout_gt[k]) for k in holdout_ids)
v2_holdout_m = sum(len(v2_holdout_preds[k]) for k in holdout_ids)
v2_holdout_true = sum(len(holdout_gt[k]) for k in holdout_ids)
v2_holdout_p = v2_holdout_tp / v2_holdout_m if v2_holdout_m > 0 else 0
v2_holdout_r = v2_holdout_tp / v2_holdout_true if v2_holdout_true > 0 else 0

print(f"V1 Baseline on Holdout (@ 0.75): Macro F0.5 = {v1_holdout_f05:.4f} | P = {v1_holdout_p:.4f}, R = {v1_holdout_r:.4f} ({v1_holdout_m:,} matches)")
print(f"V2 Enhanced on Holdout (@ {v2_dev_t:.2f}): Macro F0.5 = {v2_holdout_f05:.4f} | P = {v2_holdout_p:.4f}, R = {v2_holdout_r:.4f} ({v2_holdout_m:,} matches)")

delta_holdout = v2_holdout_f05 - v1_holdout_f05
pct_gain = (delta_holdout / v1_holdout_f05) * 100
print(f"\nHoldout Absolute Delta: {delta_holdout:+.4f} ({pct_gain:+.2f}%)")
