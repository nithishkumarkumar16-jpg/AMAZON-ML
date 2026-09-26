"""Fast, single-pass training validation script for baseline model.

Samples 5,000 S1 validation entities (3,000 US + 2,000 India), streams S2 and S3
in a single pass, computes recall and macro F_0.5 across thresholds, and selects
the optimal conservative threshold.
"""

import os
import sys
import time
from collections import defaultdict
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

SAMPLE_SIZE = 5_000  # 3,000 US + 2,000 India

print(f"=== Running Training Validation on {SAMPLE_SIZE:,} S1 Entities ===", flush=True)

# 1. Load S1 Sample
print("Loading train_source1 sample...", flush=True)
s1_df = load_source_tsv(TRAIN_SOURCE1, nrows=SAMPLE_SIZE * 3)

s1_us = s1_df[s1_df["country"] == "US"].head(int(SAMPLE_SIZE * 0.6))
s1_in = s1_df[s1_df["country"] == "India"].head(int(SAMPLE_SIZE * 0.4))
s1_val = pd.concat([s1_us, s1_in], ignore_index=True)
print(f"Sampled {len(s1_val):,} S1 entities: {len(s1_us):,} US, {len(s1_in):,} India", flush=True)

# Pre-normalize S1 and collect target keys per country
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
print(f"Target blocking keys: {len(target_keys['US']):,} US, {len(target_keys['India']):,} India", flush=True)

# 2. Load Ground Truth for validation sample
print("Loading ground truth for validation entities...", flush=True)
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

total_true_matches = sum(len(s) for s in gt_records.values())
singletons = sum(1 for s in gt_records.values() if len(s) == 0)
print(f"Loaded ground truth: {total_true_matches:,} true match pairs, {singletons:,} singletons ({singletons/len(gt_records)*100:.2f}%)", flush=True)

# 3. Build Blocking Indexes in a Single Pass over S2 and S3
indexes = {"US": BlockingIndex(max_bucket_size=150), "India": BlockingIndex(max_bucket_size=150)}
cand_store = {}  # cid -> (norm_name, norm_addr)

# Stream S2
t0 = time.time()
print("Streaming train_source2.tsv in single pass...", flush=True)
with open(TRAIN_SOURCE2, "r", encoding="utf-8") as f:
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

print(f"S2 processed in {time.time() - t0:.2f}s. Stored candidates so far: {len(cand_store):,}", flush=True)

# Stream S3
t0 = time.time()
print("Streaming train_source3.tsv in single pass...", flush=True)
with open(TRAIN_SOURCE3, "r", encoding="utf-8") as f:
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

print(f"S3 processed in {time.time() - t0:.2f}s. Total stored candidates: {len(cand_store):,}", flush=True)

for c in ["US", "India"]:
    indexes[c].filter_oversized_buckets()

# 4. Generate Candidate Sets
candidates_by_s1 = {}
total_candidates_generated = 0
true_matches_retrieved = 0

for s1_id, (_, s1_name, s1_addr, c) in s1_records.items():
    idx = indexes.get(c)
    c_ids = idx.get_candidates_for_name(s1_name) if idx else set()
    c_records = [(cid, cand_store[cid][0], cand_store[cid][1]) for cid in c_ids if cid in cand_store]
    candidates_by_s1[s1_id] = c_records
    total_candidates_generated += len(c_records)
    
    true_set = gt_records.get(s1_id, set())
    for cid in c_ids:
        if cid in true_set:
            true_matches_retrieved += 1

print(f"\n=== Blocking Summary ===", flush=True)
print(f"Total S1 entities: {len(s1_records):,}")
print(f"Total candidates generated: {total_candidates_generated:,} (avg {total_candidates_generated/len(s1_records):.2f} per entity)")
print(f"True matches retrieved in candidates: {true_matches_retrieved:,} / {total_true_matches:,} ({true_matches_retrieved/total_true_matches*100:.2f}% recall ceiling)")

# 5. Evaluate Macro F_0.5 across conservative thresholds
thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
print(f"\n=== Evaluating Macro F_0.5 Across Thresholds ===", flush=True)

best_threshold = None
best_f05 = -1.0

for thresh in thresholds:
    matcher = BaselineMatcher(threshold=thresh)
    preds = {}
    total_preds = 0
    
    for s1_id, (_, s1_name, s1_addr, _) in s1_records.items():
        cands = candidates_by_s1[s1_id]
        _, matched_ids = matcher.match_candidates(s1_name, s1_addr, cands)
        preds[s1_id] = set(matched_ids)
        total_preds += len(matched_ids)
        
    f05 = compute_f05_score(gt_records, preds)
    print(f"Threshold: {thresh:.2f} | Macro F_0.5: {f05:.4f} | Total Predicted Matches: {total_preds:,}")
    if f05 > best_f05:
        best_f05 = f05
        best_threshold = thresh

print(f"\n>>> Selected Best Threshold: {best_threshold:.2f} with Macro F_0.5 = {best_f05:.4f} <<<", flush=True)
