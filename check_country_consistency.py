import os
import csv
import sys

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

dataset_dir = r"d:\AMAZON ML\dataset"

# Load country of S1
print("Checking country consistency across ground truth...", flush=True)
s1_country = {}
with open(os.path.join(dataset_dir, "train", "train_source1.tsv"), "r", encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        s1_country[row[0].strip()] = row[3].strip()

# Load country of S2
s2_country = {}
with open(os.path.join(dataset_dir, "train", "train_source2.tsv"), "r", encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        s2_country[row[0].strip()] = row[3].strip()

# Load country of S3
s3_country = {}
with open(os.path.join(dataset_dir, "train", "train_source3.tsv"), "r", encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        s3_country[row[0].strip()] = row[3].strip()

cross_country_matches = 0
total_checked = 0

with open(os.path.join(dataset_dir, "train", "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for row in r:
        s1_id = row[0].strip()
        c1 = s1_country.get(s1_id)
        m_val = row[1].strip() if len(row) > 1 else ""
        if not m_val:
            continue
        m_ids = [m.strip() for m in m_val.split(",") if m.strip()]
        for mid in m_ids:
            total_checked += 1
            c2 = s2_country.get(mid) or s3_country.get(mid)
            if c1 != c2:
                cross_country_matches += 1
                if cross_country_matches <= 10:
                    print(f"Cross-country match! S1: {s1_id} ({c1}) vs Match: {mid} ({c2})")

print(f"\nTotal matches checked: {total_checked:,}")
print(f"Cross-country matches: {cross_country_matches} ({(cross_country_matches/total_checked)*100:.6f}%)")
