import csv
import os
import sys
from collections import Counter, defaultdict

dataset_dir = r"d:\AMAZON ML\dataset"

files = {
    "train_s1": os.path.join(dataset_dir, "train", "train_source1.tsv"),
    "train_s2": os.path.join(dataset_dir, "train", "train_source2.tsv"),
    "train_s3": os.path.join(dataset_dir, "train", "train_source3.tsv"),
    "train_gt": os.path.join(dataset_dir, "train", "train_ground_truth.tsv"),
    "test_s1": os.path.join(dataset_dir, "test", "test_source1.tsv"),
    "test_s2": os.path.join(dataset_dir, "test", "test_source2.tsv"),
    "test_s3": os.path.join(dataset_dir, "test", "test_source3.tsv"),
}

print("=== 1. ROW COUNTS & 2. EXACT COLUMN NAMES & 3. MISSING VALUES & 4. DUPLICATE IDS ===")
source_stats = {}

for label, filepath in files.items():
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        continue
    
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        
        row_count = 0
        missing_counts = {col: 0 for col in header}
        ids = set()
        duplicate_id_count = 0
        country_counts = Counter()
        
        name_lens = []
        name_words = []
        addr_lens = []
        addr_words = []
        
        # Ground truth specific
        match_lens = []
        matched_s2_count = 0
        matched_s3_count = 0
        
        for row in reader:
            row_count += 1
            if len(row) < len(header):
                # Row has fewer columns than header
                for i in range(len(header)):
                    val = row[i] if i < len(row) else ""
                    if not val.strip():
                        missing_counts[header[i]] += 1
            else:
                for i, col in enumerate(header):
                    if not row[i].strip():
                        missing_counts[col] += 1
            
            # ID check
            if len(row) > 0:
                ent_id = row[0].strip()
                if ent_id in ids:
                    duplicate_id_count += 1
                else:
                    ids.add(ent_id)
            
            # Country and text stats if source file
            if "country" in header:
                c_idx = header.index("country")
                c_val = row[c_idx].strip() if len(row) > c_idx else ""
                country_counts[c_val] += 1
            
            if "business_name" in header:
                n_idx = header.index("business_name")
                name_val = row[n_idx].strip() if len(row) > n_idx else ""
                name_lens.append(len(name_val))
                name_words.append(len(name_val.split()) if name_val else 0)
                
            if "business_address" in header:
                a_idx = header.index("business_address")
                addr_val = row[a_idx].strip() if len(row) > a_idx else ""
                addr_lens.append(len(addr_val))
                addr_words.append(len(addr_val.split()) if addr_val else 0)
                
            if "matched_entity_ids" in header:
                m_idx = header.index("matched_entity_ids")
                m_val = row[m_idx].strip() if len(row) > m_idx else ""
                if m_val:
                    m_ids = [m.strip() for m in m_val.split(",") if m.strip()]
                    match_lens.append(len(m_ids))
                    for mid in m_ids:
                        if mid.startswith("S2-"):
                            matched_s2_count += 1
                        elif mid.startswith("S3-"):
                            matched_s3_count += 1
                else:
                    match_lens.append(0)
                    
        print(f"\n--- {label} ({os.path.basename(filepath)}) ---")
        print(f"Total Rows: {row_count:,}")
        print(f"Exact Header: {header}")
        print(f"Missing Values: {missing_counts}")
        print(f"Unique IDs: {len(ids):,}, Duplicate IDs: {duplicate_id_count}")
        if country_counts:
            print(f"Country Counts: {dict(country_counts)}")
        if name_lens:
            print(f"Name Length (chars) - min: {min(name_lens)}, max: {max(name_lens)}, avg: {sum(name_lens)/len(name_lens):.2f}")
            print(f"Name Words - min: {min(name_words)}, max: {max(name_words)}, avg: {sum(name_words)/len(name_words):.2f}")
        if addr_lens:
            print(f"Address Length (chars) - min: {min(addr_lens)}, max: {max(addr_lens)}, avg: {sum(addr_lens)/len(addr_lens):.2f}")
            print(f"Address Words - min: {min(addr_words)}, max: {max(addr_words)}, avg: {sum(addr_words)/len(addr_words):.2f}")
            
        if label == "train_gt":
            zero_matches = sum(1 for x in match_lens if x == 0)
            one_match = sum(1 for x in match_lens if x == 1)
            multi_matches = sum(1 for x in match_lens if x > 1)
            total_matches = sum(match_lens)
            print("\n--- Ground Truth Match Stats ---")
            print(f"Total S1 entities in GT: {len(match_lens):,}")
            print(f"Total matches referenced: {total_matches:,}")
            print(f"Matched S2 references: {matched_s2_count:,} ({matched_s2_count/total_matches*100:.2f}%)")
            print(f"Matched S3 references: {matched_s3_count:,} ({matched_s3_count/total_matches*100:.2f}%)")
            print(f"Zero matches (singletons): {zero_matches:,} ({zero_matches/len(match_lens)*100:.2f}%)")
            print(f"One match: {one_match:,} ({one_match/len(match_lens)*100:.2f}%)")
            print(f"Multiple matches (>1): {multi_matches:,} ({multi_matches/len(match_lens)*100:.2f}%)")
            print(f"Max matches for a single S1: {max(match_lens)}")
            print(f"Distribution of match counts: {Counter(match_lens).most_common(15)}")
