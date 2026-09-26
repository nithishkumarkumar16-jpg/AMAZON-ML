import os
import sys
import time
from collections import Counter, defaultdict

dataset_dir = r"d:\AMAZON ML\dataset"

files = [
    ("train_source1.tsv", os.path.join(dataset_dir, "train", "train_source1.tsv")),
    ("train_source2.tsv", os.path.join(dataset_dir, "train", "train_source2.tsv")),
    ("train_source3.tsv", os.path.join(dataset_dir, "train", "train_source3.tsv")),
    ("train_ground_truth.tsv", os.path.join(dataset_dir, "train", "train_ground_truth.tsv")),
    ("test_source1.tsv", os.path.join(dataset_dir, "test", "test_source1.tsv")),
    ("test_source2.tsv", os.path.join(dataset_dir, "test", "test_source2.tsv")),
    ("test_source3.tsv", os.path.join(dataset_dir, "test", "test_source3.tsv")),
]

results = {}

print("Starting fast dataset inspection...", flush=True)

for fname, fpath in files:
    t0 = time.time()
    print(f"\nProcessing {fname}...", flush=True)
    if not os.path.exists(fpath):
        print(f"File not found: {fpath}", flush=True)
        continue

    with open(fpath, "r", encoding="utf-8") as f:
        header_line = f.readline()
        header = [c.strip() for c in header_line.rstrip("\r\n").split("\t")]
        num_cols = len(header)
        
        row_count = 0
        missing_counts = {col: 0 for col in header}
        
        ids_set = set()
        dup_id_count = 0
        
        country_counts = Counter()
        
        name_count = 0
        name_len_sum = 0
        name_min_len = 999999
        name_max_len = 0
        name_word_sum = 0
        name_min_words = 999999
        name_max_words = 0
        
        addr_count = 0
        addr_len_sum = 0
        addr_min_len = 999999
        addr_max_len = 0
        addr_word_sum = 0
        addr_min_words = 999999
        addr_max_words = 0
        
        # Ground truth specific
        match_distribution = Counter()
        total_s2_matches = 0
        total_s3_matches = 0
        max_matches = 0
        total_gt_matches = 0
        
        col_indices = {col: i for i, col in enumerate(header)}
        
        for line in f:
            line_clean = line.rstrip("\r\n")
            if not line_clean:
                continue
            row_count += 1
            parts = line_clean.split("\t")
            
            # Check ID
            ent_id = parts[0].strip()
            if ent_id in ids_set:
                dup_id_count += 1
            else:
                ids_set.add(ent_id)
                
            # Check missing
            for col, idx in col_indices.items():
                if idx >= len(parts) or not parts[idx].strip():
                    missing_counts[col] += 1
            
            # Country
            if "country" in col_indices:
                c_idx = col_indices["country"]
                c_val = parts[c_idx].strip() if c_idx < len(parts) else ""
                country_counts[c_val] += 1
                
            # Name
            if "business_name" in col_indices:
                n_idx = col_indices["business_name"]
                n_val = parts[n_idx].strip() if n_idx < len(parts) else ""
                if n_val:
                    n_len = len(n_val)
                    n_words = len(n_val.split())
                    name_count += 1
                    name_len_sum += n_len
                    name_word_sum += n_words
                    if n_len < name_min_len: name_min_len = n_len
                    if n_len > name_max_len: name_max_len = n_len
                    if n_words < name_min_words: name_min_words = n_words
                    if n_words > name_max_words: name_max_words = n_words
                    
            # Address
            if "business_address" in col_indices:
                a_idx = col_indices["business_address"]
                a_val = parts[a_idx].strip() if a_idx < len(parts) else ""
                if a_val:
                    a_len = len(a_val)
                    a_words = len(a_val.split())
                    addr_count += 1
                    addr_len_sum += a_len
                    addr_word_sum += a_words
                    if a_len < addr_min_len: addr_min_len = a_len
                    if a_len > addr_max_len: addr_max_len = a_len
                    if a_words < addr_min_words: addr_min_words = a_words
                    if a_words > addr_max_words: addr_max_words = a_words
                    
            # Ground truth
            if "matched_entity_ids" in col_indices:
                m_idx = col_indices["matched_entity_ids"]
                m_val = parts[m_idx].strip() if m_idx < len(parts) else ""
                if not m_val:
                    match_distribution[0] += 1
                else:
                    m_ids = [m.strip() for m in m_val.split(",") if m.strip()]
                    num_m = len(m_ids)
                    match_distribution[num_m] += 1
                    total_gt_matches += num_m
                    if num_m > max_matches: max_matches = num_m
                    for mid in m_ids:
                        if mid.startswith("S2-"):
                            total_s2_matches += 1
                        elif mid.startswith("S3-"):
                            total_s3_matches += 1

        dt = time.time() - t0
        print(f"Finished {fname} in {dt:.2f}s", flush=True)
        print(f"  Rows: {row_count:,}", flush=True)
        print(f"  Exact Columns: {header}", flush=True)
        print(f"  Missing values: {missing_counts}", flush=True)
        print(f"  Unique IDs: {len(ids_set):,}, Duplicate IDs: {dup_id_count}", flush=True)
        if country_counts:
            print(f"  Country breakdown: {dict(country_counts)}", flush=True)
        if name_count > 0:
            print(f"  Name char length: min={name_min_len}, max={name_max_len}, avg={name_len_sum/name_count:.2f}", flush=True)
            print(f"  Name word count: min={name_min_words}, max={name_max_words}, avg={name_word_sum/name_count:.2f}", flush=True)
        if addr_count > 0:
            print(f"  Address char length: min={addr_min_len}, max={addr_max_len}, avg={addr_len_sum/addr_count:.2f}", flush=True)
            print(f"  Address word count: min={addr_min_words}, max={addr_max_words}, avg={addr_word_sum/addr_count:.2f}", flush=True)
            
        if fname == "train_ground_truth.tsv":
            n_entities = row_count
            n_zero = match_distribution[0]
            n_one = match_distribution[1]
            n_multi = sum(count for k, count in match_distribution.items() if k > 1)
            print(f"  Ground Truth Evaluation:", flush=True)
            print(f"    Total S1 rows in GT: {n_entities:,}", flush=True)
            print(f"    Total referenced match pairs: {total_gt_matches:,}", flush=True)
            print(f"    Matches from S2: {total_s2_matches:,} ({total_s2_matches/total_gt_matches*100:.2f}%)", flush=True)
            print(f"    Matches from S3: {total_s3_matches:,} ({total_s3_matches/total_gt_matches*100:.2f}%)", flush=True)
            print(f"    Zero matches (singletons): {n_zero:,} ({n_zero/n_entities*100:.2f}%)", flush=True)
            print(f"    Exactly one match: {n_one:,} ({n_one/n_entities*100:.2f}%)", flush=True)
            print(f"    Multiple matches (>1): {n_multi:,} ({n_multi/n_entities*100:.2f}%)", flush=True)
            print(f"    Max matches for a single entity: {max_matches}", flush=True)
            print(f"    Match count distribution (k: count): {sorted(match_distribution.items())[:15]}", flush=True)

print("\n=== ALL FILES PROCESSED SUCCESSFULLY ===", flush=True)
