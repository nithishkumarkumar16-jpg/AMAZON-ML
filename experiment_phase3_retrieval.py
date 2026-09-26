import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
src_dir = Path("code/business_entity_resolution/src").resolve()
sys.path.insert(0, str(src_dir))

from normalization import normalize_business_name, normalize_business_address, normalize_country
from predict_v2 import get_v2_blocking_keys

print("=" * 65)
print("PHASE 3: COMPLEMENTARY RETRIEVAL EXPERIMENT ON 1,000 DEV ENTITIES")
print("=" * 65)

# 1. Helper to extract address blocking keys
def get_address_blocking_keys(norm_addr: str) -> list:
    """Extract precision-calibrated address keys.
    
    Format:
    addr:<door_number>_<first_street_word>
    Requires both a numeric token and a street token of length >= 3.
    """
    if not norm_addr:
        return []
    
    tokens = norm_addr.split()
    nums = [t for t in tokens if any(c.isdigit() for c in t)]
    words = [t for t in tokens if not any(c.isdigit() for c in t) and len(t) >= 3 and t not in {
        "street", "road", "avenue", "lane", "drive", "court", "circle", "boulevard",
        "near", "opp", "opposite", "plot", "number", "door", "floor", "building", "phase"
    }]
    
    keys = []
    if nums and words:
        primary_num = nums[0]
        primary_word = words[0]
        keys.append(f"addr:{primary_num}_{primary_word}")
        if len(words) >= 2:
            keys.append(f"addr:{primary_num}_{words[1]}")
    return keys

# 2. Phase 3 combined blocking keys
def get_phase3_blocking_keys(norm_name: str, norm_addr: str) -> list:
    keys = get_v2_blocking_keys(norm_name)
    keys.extend(get_address_blocking_keys(norm_addr))
    return list(dict.fromkeys(keys))

# 3. Load 1,000 Dev entities
dev_s1 = {}
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
        n = normalize_business_name(parts[1])
        a = normalize_business_address(parts[2])
        dev_s1[sid] = (sid, n, a, c)
        country_counts[c] += 1

dev_ids = set(dev_s1.keys())

# Load ground truth
dev_gt = {}
with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
    f.readline()
    for line in f:
        parts = line.rstrip("\r\n").split("\t")
        sid = parts[0].strip()
        if sid in dev_ids:
            m_val = parts[1] if len(parts) > 1 else ""
            m_ids = {m.strip() for m in m_val.split(",") if m.strip()} if m_val else set()
            dev_gt[sid] = m_ids
            if len(dev_gt) == len(dev_ids): break

total_gt = sum(len(s) for s in dev_gt.values())
print(f"Sample Size:            {len(dev_s1):,} S1 entities ({dict(country_counts)})")
print(f"Total True Matches:     {total_gt:,} true match pairs")

# Target keys for both strategies
target_v2_keys = {"US": set(), "INDIA": set()}
target_p3_keys = {"US": set(), "INDIA": set()}

for sid, (_, n, a, c) in dev_s1.items():
    v2_k = get_v2_blocking_keys(n)
    p3_k = get_phase3_blocking_keys(n, a)
    target_v2_keys[c].update(v2_k)
    target_p3_keys[c].update(p3_k)

print(f"V2 Target Keys:  US={len(target_v2_keys['US']):,}, INDIA={len(target_v2_keys['INDIA']):,}")
print(f"P3 Target Keys:  US={len(target_p3_keys['US']):,}, INDIA={len(target_p3_keys['INDIA']):,}")

# Build inverted indexes for both strategies across S2 and S3
v2_index = {"US": defaultdict(list), "INDIA": defaultdict(list)}
p3_index = {"US": defaultdict(list), "INDIA": defaultdict(list)}
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
            if c not in {"US", "INDIA"}: continue
            
            t2_keys = target_v2_keys[c]
            t3_keys = target_p3_keys[c]
            
            cname = parts[1]
            caddr = parts[2]
            cid = parts[0].strip()
            norm_n = normalize_business_name(cname)
            norm_a = normalize_business_address(caddr)
            
            v2_k = get_v2_blocking_keys(norm_n)
            p3_k = get_phase3_blocking_keys(norm_n, norm_a)
            
            m_v2 = [k for k in v2_k if k in t2_keys]
            m_p3 = [k for k in p3_k if k in t3_keys]
            
            if m_v2 or m_p3:
                cand_store[cid] = (norm_n, norm_a)
                for k in m_v2:
                    v2_index[c][k].append(cid)
                for k in m_p3:
                    p3_index[c][k].append(cid)
                    
    print(f"  {label} processed in {time.time() - t0:.2f}s. Loaded {len(cand_store):,} distinct candidates.", flush=True)

# Filter oversized buckets
for c in ["US", "INDIA"]:
    oversized_v2 = [k for k, v in v2_index[c].items() if len(v) > MAX_BUCKET]
    for k in oversized_v2: del v2_index[c][k]
    
    oversized_p3 = [k for k, v in p3_index[c].items() if len(v) > MAX_BUCKET]
    for k in oversized_p3: del p3_index[c][k]

# Evaluate Candidate Retrieval for V2 vs Phase 3
def evaluate_retrieval(strategy_name, index_dict, key_fn):
    retrieved_true = 0
    cands_per_s1 = []
    recovered_pairs = set()
    
    for sid, (_, n, a, c) in dev_s1.items():
        keys = key_fn(n, a)
        c_ids = set()
        c_idx = index_dict[c]
        for k in keys:
            if k in c_idx:
                c_ids.update(c_idx[k])
        cands_per_s1.append(len(c_ids))
        true_matches = c_ids & dev_gt[sid]
        retrieved_true += len(true_matches)
        for cid in true_matches:
            recovered_pairs.add((sid, cid))
            
    recall = retrieved_true / total_gt * 100
    total_cands = sum(cands_per_s1)
    avg_cands = np.mean(cands_per_s1)
    p95_cands = np.percentile(cands_per_s1, 95)
    p99_cands = np.percentile(cands_per_s1, 99)
    max_cands = max(cands_per_s1)
    
    print(f"\n--- {strategy_name} ---")
    print(f"  Candidate Recall:       {retrieved_true:,} / {total_gt:,} ({recall:.2f}%)")
    print(f"  Total Candidates:       {total_cands:,}")
    print(f"  Average Cands / S1:     {avg_cands:.2f}")
    print(f"  p95 Cands / S1:         {p95_cands:.1f}")
    print(f"  p99 Cands / S1:         {p99_cands:.1f}")
    print(f"  Max Cands for single S1:{max_cands:,}")
    return recovered_pairs, retrieved_true, total_cands, avg_cands, p95_cands

v2_pairs, v2_rec, v2_tot, v2_avg, v2_p95 = evaluate_retrieval("Strategy A: V2 Baseline (Name Blocking only)", v2_index, lambda n, a: get_v2_blocking_keys(n))
p3_pairs, p3_rec, p3_tot, p3_avg, p3_p95 = evaluate_retrieval("Strategy B: Phase 3 (Name + Address-Aware Blocking)", p3_index, get_phase3_blocking_keys)

newly_recovered = p3_pairs - v2_pairs
lost_pairs = v2_pairs - p3_pairs

print("\n" + "=" * 65)
print("HEAD-TO-HEAD CANDIDATE RETRIEVAL SUMMARY")
print("=" * 65)
print(f"Baseline Recall (V2):      {v2_rec:,} / {total_gt:,} ({v2_rec/total_gt*100:.2f}%)")
print(f"Phase 3 Recall:            {p3_rec:,} / {total_gt:,} ({p3_rec/total_gt*100:.2f}%)")
print(f"Absolute Recall Delta:     {p3_rec - v2_rec:+,} true pairs (+{(p3_rec - v2_rec)/total_gt*100:.2f}% abs recall!)")
print(f"Newly Recovered Pairs:     +{len(newly_recovered):,} true pairs")
print(f"Lost Pairs:                -{len(lost_pairs):,} pairs")
print(f"Candidate Growth Ratio:    {p3_tot / v2_tot:.2f}x ({p3_tot:,} vs {v2_tot:,})")
print(f"Average Candidate Count:   {p3_avg:.1f} vs {v2_avg:.1f} per S1 (p95: {p3_p95:.1f} vs {v2_p95:.1f})")

if newly_recovered:
    print("\nSample Newly Recovered True Pairs:")
    for sid, cid in list(newly_recovered)[:5]:
        _, s1_n, s1_a, _ = dev_s1[sid]
        cn, ca = cand_store[cid]
        print(f"  S1:   '{s1_n}' | '{s1_a}'")
        print(f"  Cand: '{cn}' | '{ca}'\n")
