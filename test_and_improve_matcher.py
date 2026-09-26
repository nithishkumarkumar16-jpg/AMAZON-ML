"""Phase 2 — Evaluator and Optimizer for Entity Matching on Training Ground Truth.

Evaluates the baseline and candidate improvements across:
Thresholds: 0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.85
Reports: Macro F_0.5, Macro Precision, Macro Recall, and Total Matches.
"""

import os
import sys
import time
import pickle
import difflib
from pathlib import Path
import pandas as pd

# Setup package paths
src_dir = Path(__file__).resolve().parent / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(src_dir))

from config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
)
from normalization import (
    normalize_business_name,
    normalize_business_address,
    normalize_country,
)
from blocking import BlockingIndex, get_blocking_keys
from features import compute_pair_features, score_candidate_pair
from matcher import BaselineMatcher
from evaluation import compute_f05_score
from data_loader import load_source_tsv

CACHE_FILE = Path(__file__).resolve().parent / "output" / "val_cache_5k.pkl"
SAMPLE_SIZE = 5_000  # 3,000 US + 2,000 India


def compute_metrics(ground_truth, predictions):
    """Compute Macro F0.5, Macro Precision, and Macro Recall across all entities."""
    total_f05 = 0.0
    total_p = 0.0
    total_r = 0.0
    num_entities = len(ground_truth)

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())

        if len(true_set) == 0:
            if len(pred_set) == 0:
                p, r, f05 = 1.0, 1.0, 1.0
            else:
                p, r, f05 = 0.0, 0.0, 0.0
        else:
            if len(pred_set) == 0:
                p, r, f05 = 0.0, 0.0, 0.0
            else:
                tp = len(true_set & pred_set)
                if tp == 0:
                    p, r, f05 = 0.0, 0.0, 0.0
                else:
                    p = tp / len(pred_set)
                    r = tp / len(true_set)
                    denom = 0.25 * p + r
                    f05 = (1.25 * p * r) / denom if denom > 0 else 0.0

        total_f05 += f05
        total_p += p
        total_r += r

    return (
        total_f05 / num_entities,
        total_p / num_entities,
        total_r / num_entities,
    )


def get_validation_data():
    """Load or generate cached validation data."""
    if CACHE_FILE.exists():
        print(f"Loading cached validation split from {CACHE_FILE}...", flush=True)
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    print("Generating validation split...", flush=True)
    s1_df = load_source_tsv(TRAIN_SOURCE1, nrows=SAMPLE_SIZE * 3)
    s1_us = s1_df[s1_df["country"] == "US"].head(int(SAMPLE_SIZE * 0.6))
    s1_in = s1_df[s1_df["country"] == "India"].head(int(SAMPLE_SIZE * 0.4))
    s1_val = pd.concat([s1_us, s1_in], ignore_index=True)

    s1_records = {}
    target_keys = {"US": set(), "India": set()}
    for _, row in s1_val.iterrows():
        s1_id = row["entity_id"]
        c = row["country"].strip()
        n = normalize_business_name(row["business_name"])
        a = normalize_business_address(row["business_address"])
        s1_records[s1_id] = (s1_id, n, a, c)
        if c in target_keys:
            target_keys[c].update(get_blocking_keys(n))

    val_ids = set(s1_records.keys())

    # Ground truth
    gt_records = {}
    with open(TRAIN_GROUND_TRUTH, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            if s1_id in val_ids:
                m_val = parts[1] if len(parts) > 1 else ""
                m_ids = {m.strip() for m in m_val.split(",") if m.strip()} if m_val else set()
                gt_records[s1_id] = m_ids
                if len(gt_records) == len(val_ids):
                    break

    # Stream S2 & S3
    indexes = {"US": BlockingIndex(max_bucket_size=150), "India": BlockingIndex(max_bucket_size=150)}
    cand_store = {}

    for s_file, name in [(TRAIN_SOURCE2, "S2"), (TRAIN_SOURCE3, "S3")]:
        t0 = time.time()
        print(f"Streaming {name}...", flush=True)
        with open(s_file, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4: continue
                c = parts[3].strip()
                tkeys = target_keys.get(c)
                if not tkeys: continue

                cname = parts[1]
                norm_n = normalize_business_name(cname)
                keys = get_blocking_keys(norm_n)
                matching_keys = [k for k in keys if k in tkeys]
                if matching_keys:
                    cid = parts[0]
                    norm_a = normalize_business_address(parts[2])
                    indexes[c].add_record(cid, matching_keys)
                    cand_store[cid] = (norm_n, norm_a)
        print(f"{name} done in {time.time() - t0:.2f}s.", flush=True)

    for c in ["US", "India"]:
        indexes[c].filter_oversized_buckets()

    candidates_by_s1 = {}
    for s1_id, (_, s1_name, s1_addr, c) in s1_records.items():
        idx = indexes.get(c)
        c_ids = idx.get_candidates_for_name(s1_name) if idx else set()
        c_records = [(cid, cand_store[cid][0], cand_store[cid][1]) for cid in c_ids if cid in cand_store]
        candidates_by_s1[s1_id] = c_records

    cache_data = (s1_records, gt_records, candidates_by_s1)
    os.makedirs(CACHE_FILE.parent, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache_data, f)
    print(f"Saved validation cache to {CACHE_FILE}.", flush=True)
    return cache_data


# --- FAST SIMILARITY UTILITIES FOR ENHANCED MATCHER ---

def char_ngram_dice(s1: str, s2: str, n: int = 3) -> float:
    """Compute character n-gram Dice coefficient."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    if len(s1) < n or len(s2) < n:
        return 1.0 if s1 == s2 else 0.0
    
    ngrams1 = {s1[i:i+n] for i in range(len(s1) - n + 1)}
    ngrams2 = {s2[i:i+n] for i in range(len(s2) - n + 1)}
    
    inter = len(ngrams1 & ngrams2)
    total = len(ngrams1) + len(ngrams2)
    return (2.0 * inter) / total if total > 0 else 0.0


def edit_similarity(s1: str, s2: str) -> float:
    """Compute string edit similarity via SequenceMatcher ratio."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    max_l = max(len(s1), len(s2))
    min_l = min(len(s1), len(s2))
    if min_l / max_l < 0.35:
        return 0.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def compute_enhanced_features(s1_name, s1_addr, cand_name, cand_addr):
    """Compute enriched feature set with character n-grams and edit similarity."""
    feats = {}

    # Exact equality
    exact = 1.0 if s1_name and s1_name == cand_name else 0.0
    feats["name_exact"] = exact

    # Space-stripped equality
    ns1 = s1_name.replace(" ", "")
    ns2 = cand_name.replace(" ", "")
    ns_match = 1.0 if len(ns1) >= 3 and ns1 == ns2 else 0.0
    feats["name_ns_match"] = ns_match

    # Token sets
    toks1 = set(s1_name.split())
    toks2 = set(cand_name.split())
    n_union = len(toks1 | toks2)
    n_inter = len(toks1 & toks2)
    feats["name_jaccard"] = (n_inter / n_union) if n_union > 0 else 0.0

    min_t = min(len(toks1), len(toks2))
    feats["name_containment"] = (n_inter / min_t) if min_t > 0 else 0.0

    # Character 3-gram Dice
    feats["name_char_dice3"] = char_ngram_dice(ns1, ns2, n=3)

    # Edit similarity
    feats["name_edit_sim"] = edit_similarity(s1_name, cand_name)

    # Length difference penalty
    max_len = max(len(s1_name), len(cand_name), 1)
    feats["name_len_diff"] = abs(len(s1_name) - len(cand_name)) / max_len

    # --- Address Features ---
    addr_missing = 1.0 if (not s1_addr or not cand_addr) else 0.0
    feats["addr_missing"] = addr_missing

    if addr_missing == 0.0:
        feats["addr_exact"] = 1.0 if s1_addr == cand_addr else 0.0
        
        atok1 = set(s1_addr.split())
        atok2 = set(cand_addr.split())
        a_union = len(atok1 | atok2)
        a_inter = len(atok1 & atok2)
        feats["addr_jaccard"] = (a_inter / a_union) if a_union > 0 else 0.0

        min_a = min(len(atok1), len(atok2))
        feats["addr_containment"] = (a_inter / min_a) if min_a > 0 else 0.0

        # Character 3-gram Dice on address
        feats["addr_char_dice3"] = char_ngram_dice(s1_addr.replace(" ", ""), cand_addr.replace(" ", ""), n=3)

        # Numbers
        nums1 = {t for t in atok1 if any(c.isdigit() for c in t)}
        nums2 = {t for t in atok2 if any(c.isdigit() for c in t)}
        if nums1 and nums2:
            num_inter = len(nums1 & nums2)
            num_union = len(nums1 | nums2)
            feats["numeric_jaccard"] = num_inter / num_union
            feats["numeric_conflict"] = 1.0 if num_inter == 0 else 0.0
        elif not nums1 and not nums2:
            feats["numeric_jaccard"] = 1.0
            feats["numeric_conflict"] = 0.0
        else:
            feats["numeric_jaccard"] = 0.0
            feats["numeric_conflict"] = 0.0
    else:
        feats["addr_exact"] = 0.0
        feats["addr_jaccard"] = 0.0
        feats["addr_containment"] = 0.0
        feats["addr_char_dice3"] = 0.0
        feats["numeric_jaccard"] = 0.0
        feats["numeric_conflict"] = 0.0

    return feats


def score_enhanced_pair(feats):
    """Enhanced scoring combining exact, token, character n-gram, edit, and address."""
    # Multi-signal name score
    name_score = max(
        feats["name_exact"],
        feats["name_ns_match"],
        feats["name_jaccard"] * 0.95,
        feats["name_containment"] * 0.85,
        feats["name_char_dice3"] * 0.92,
        feats["name_edit_sim"] * 0.90,
    )

    # Length penalty for non-exact matches
    if feats["name_exact"] == 0.0 and feats["name_ns_match"] == 0.0:
        name_score -= feats["name_len_diff"] * 0.12

    name_score = max(0.0, min(1.0, name_score))

    if feats["addr_missing"] == 0.0:
        if feats.get("numeric_conflict", 0.0) == 1.0:
            addr_score = feats["addr_jaccard"] * 0.10
            composite = (0.50 * name_score + 0.50 * addr_score) * 0.65
        else:
            addr_score = max(
                feats["addr_exact"],
                0.40 * feats["addr_jaccard"] + 0.35 * feats["addr_containment"] + 0.25 * feats["numeric_jaccard"],
                feats["addr_char_dice3"] * 0.85,
            )
            composite = 0.60 * name_score + 0.40 * addr_score
    else:
        if name_score >= 0.85:
            composite = name_score * 0.92
        elif name_score >= 0.75:
            composite = name_score * 0.82
        else:
            composite = name_score * 0.60

    return max(0.0, min(1.0, composite))


def main():
    s1_records, gt_records, candidates_by_s1 = get_validation_data()
    eval_thresholds = [0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.82, 0.85]

    print("\n" + "=" * 75)
    print("1. EVALUATING CURRENT BASELINE MATCHER")
    print("=" * 75)
    print(f"{'Threshold':<10} | {'Macro F0.5':<12} | {'Precision':<12} | {'Recall':<12} | {'Matches':<10}")
    print("-" * 75)

    base_results = {}
    for thresh in eval_thresholds:
        preds = {}
        tot_m = 0
        for s1_id, (_, s1_name, s1_addr, _) in s1_records.items():
            matched = []
            for cid, cname, caddr in candidates_by_s1[s1_id]:
                feats = compute_pair_features(s1_name, s1_addr, cname, caddr)
                score = score_candidate_pair(feats)
                if score >= thresh:
                    matched.append(cid)
            preds[s1_id] = set(matched)
            tot_m += len(matched)

        f05, p, r = compute_metrics(gt_records, preds)
        base_results[thresh] = (f05, p, r, tot_m)
        print(f"{thresh:<10.2f} | {f05:<12.4f} | {p:<12.4f} | {r:<12.4f} | {tot_m:<10,}")

    print("\n" + "=" * 75)
    print("2. EVALUATING ENHANCED MATCHER (N-grams + Edit + Address Containment)")
    print("=" * 75)
    print(f"{'Threshold':<10} | {'Macro F0.5':<12} | {'Precision':<12} | {'Recall':<12} | {'Matches':<10}")
    print("-" * 75)

    enh_results = {}
    best_enh_f05 = -1.0
    best_enh_thresh = None

    for thresh in eval_thresholds:
        preds = {}
        tot_m = 0
        for s1_id, (_, s1_name, s1_addr, _) in s1_records.items():
            matched = []
            for cid, cname, caddr in candidates_by_s1[s1_id]:
                feats = compute_enhanced_features(s1_name, s1_addr, cname, caddr)
                score = score_enhanced_pair(feats)
                if score >= thresh:
                    matched.append(cid)
            preds[s1_id] = set(matched)
            tot_m += len(matched)

        f05, p, r = compute_metrics(gt_records, preds)
        enh_results[thresh] = (f05, p, r, tot_m)
        print(f"{thresh:<10.2f} | {f05:<12.4f} | {p:<12.4f} | {r:<12.4f} | {tot_m:<10,}")
        if f05 > best_enh_f05:
            best_enh_f05 = f05
            best_enh_thresh = thresh

    print("\n" + "=" * 75)
    print("3. COMPARISON: BASELINE vs ENHANCED")
    print("=" * 75)
    print(f"Baseline F0.5 @ 0.75: {base_results[0.75][0]:.4f} (P={base_results[0.75][1]:.4f}, R={base_results[0.75][2]:.4f})")
    print(f"Best Enhanced F0.5 @ {best_enh_thresh:.2f}: {best_enh_f05:.4f} (P={enh_results[best_enh_thresh][1]:.4f}, R={enh_results[best_enh_thresh][2]:.4f})")
    delta = best_enh_f05 - base_results[0.75][0]
    print(f"Absolute Delta: {delta:+.4f} ({delta/base_results[0.75][0]*100:+.2f}%)")


if __name__ == "__main__":
    main()
