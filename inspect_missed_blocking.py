"""Inspect true match pairs that were missed by candidate generation."""

import pickle
import sys
from pathlib import Path

src_dir = Path(__file__).resolve().parent / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(src_dir))

from config import TRAIN_SOURCE2, TRAIN_SOURCE3
from normalization import normalize_business_name
from blocking import get_blocking_keys

CACHE_FILE = Path("output/val_cache_5k.pkl")
with open(CACHE_FILE, "rb") as f:
    s1_records, gt_records, candidates_by_s1 = pickle.load(f)

# Find missed true matches
missed_pairs = []
for s1_id, (_, s1_name, s1_addr, country) in s1_records.items():
    c_records = candidates_by_s1.get(s1_id, [])
    c_ids = {cid for cid, _, _ in c_records}
    true_set = gt_records.get(s1_id, set())
    missed = true_set - c_ids
    for mid in missed:
        missed_pairs.append((s1_id, s1_name, s1_addr, mid, country))

print(f"Total missed true pairs: {len(missed_pairs):,}")

# Load raw names of these missed target IDs from S2 and S3
target_ids = {mid for _, _, _, mid, _ in missed_pairs[:300]}
target_info = {}

for s_file in [TRAIN_SOURCE2, TRAIN_SOURCE3]:
    with open(s_file, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts[0] in target_ids:
                target_info[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else "")
                if len(target_info) == len(target_ids):
                    break

print(f"\n--- SAMPLE OF 30 MISSED TRUE MATCHES ---")
for i, (s1_id, s1_name, s1_addr, mid, country) in enumerate(missed_pairs[:30]):
    if mid in target_info:
        cand_raw_name, cand_raw_addr = target_info[mid]
        cand_norm_name = normalize_business_name(cand_raw_name)
        s1_keys = get_blocking_keys(s1_name)
        cand_keys = get_blocking_keys(cand_norm_name)
        shared = set(s1_keys) & set(cand_keys)
        print(f"[{i+1}] Country: {country}")
        print(f"  S1:   Name='{s1_name}' | Keys={s1_keys}")
        print(f"  True: Name='{cand_norm_name}' (Raw: '{cand_raw_name}') | Keys={cand_keys}")
        print(f"  Shared Keys: {shared}")
        print()
