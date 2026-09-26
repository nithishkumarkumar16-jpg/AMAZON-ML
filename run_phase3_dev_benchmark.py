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
from predict_v3 import score_pair_v3, SELECTED_THRESHOLD
from evaluation import compute_f05_score

print("=" * 65)
print("PHASE 3: COMPREHENSIVE RETRIEVAL & MATCHER BENCHMARK")
print("Development Split: 1,000 S1 Entities (Lines 20,000–25,000)")
print("=" * 65)

# 1. Address blocking key function
def get_address_blocking_keys(norm_addr: str) -> list:
    if not norm_addr: return []
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

def get_p3_blocking_keys(norm_name: str, norm_addr: str) -> list:
    keys = get_v2_blocking_keys(norm_name)
    keys.extend(get_address_blocking_keys(norm_addr))
    return list(dict.fromkeys(keys))

# 2. Load 1,000 Dev S1 Entities
dev_s1 = {}
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

dev_ids = set(dev_s1.keys())

# 3. Load Ground Truth
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

all_true_cids = {cid for cids in dev_gt.values() for cid in cids}
total_gt = sum(len(s) for s in dev_gt.values())
print(f"Loaded {len(dev_s1):,} Dev S1 entities with {total_gt:,} true match pairs.")

# 4. Load True Candidate Records to evaluate retrieval capability
true_records = {}
for s_path in ["dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv"]:
    with open(s_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4: continue
            cid = parts[0].strip()
            if cid in all_true_cids:
                c = normalize_country(parts[3])
                n = normalize_business_name(parts[1])
                a = normalize_business_address(parts[2])
                true_records[cid] = (cid, n, a, c)
                if len(true_records) == len(all_true_cids): break
    if len(true_records) == len(all_true_cids): break

# 5. Measure Theoretical Retrieval Upper Bound (Key Overlap)
v1_hits = 0
v2_hits = 0
p3_hits = 0
recovered_by_addr = []

for sid, (_, s1_n, s1_a, s1_c) in dev_s1.items():
    k_v1 = set(get_blocking_keys(s1_n))
    k_v2 = set(get_v2_blocking_keys(s1_n))
    k_p3 = set(get_p3_blocking_keys(s1_n, s1_a))
    
    for cid in dev_gt[sid]:
        if cid not in true_records: continue
        _, cn, ca, _ = true_records[cid]
        c_k_v1 = set(get_blocking_keys(cn))
        c_k_v2 = set(get_v2_blocking_keys(cn))
        c_k_p3 = set(get_p3_blocking_keys(cn, ca))
        
        has_v1 = bool(k_v1 & c_k_v1)
        has_v2 = bool(k_v2 & c_k_v2)
        has_p3 = bool(k_p3 & c_k_p3)
        
        if has_v1: v1_hits += 1
        if has_v2: v2_hits += 1
        if has_p3: p3_hits += 1
        
        if has_p3 and not has_v2:
            recovered_by_addr.append((sid, s1_n, s1_a, cid, cn, ca))

print("\n--- RETRIEVAL KEY RECALL COMPARISON ---")
print(f"V1 (Baseline):            {v1_hits:,} / {total_gt:,} ({v1_hits/total_gt*100:.2f}%)")
print(f"V2 (Multi-Angle Name):    {v2_hits:,} / {total_gt:,} ({v2_hits/total_gt*100:.2f}%)")
print(f"Phase 3 (Name + Address): {p3_hits:,} / {total_gt:,} ({p3_hits/total_gt*100:.2f}%)")
print(f"Phase 3 Net Gain over V2: +{p3_hits - v2_hits:,} true pairs (+{(p3_hits - v2_hits)/total_gt*100:.2f}% abs recall!)")

print(f"\n--- SAMPLE NEWLY RECOVERED TRUE PAIRS VIA ADDRESS BLOCKING ---")
for s1_id, s1_n, s1_a, cid, cn, ca in recovered_by_addr[:5]:
    score = score_pair_v3(s1_n, s1_a, cn, ca)
    print(f"  S1:   '{s1_n}' | '{s1_a}'")
    print(f"  Cand: '{cn}' | '{ca}'")
    print(f"  Score: {score:.4f} (Meets threshold {SELECTED_THRESHOLD}: {score >= SELECTED_THRESHOLD})\n")
