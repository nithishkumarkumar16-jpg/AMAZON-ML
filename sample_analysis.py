import os
import csv
import sys

sys.stdout.reconfigure(encoding='utf-8')

dataset_dir = r"d:\AMAZON ML\dataset"

# Load first 20 GT pairs with matches
gt_path = os.path.join(dataset_dir, "train", "train_ground_truth.tsv")
s1_needed = {}
all_matched_ids = set()

sample_gt = []
with open(gt_path, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    header = next(reader)
    count = 0
    for row in reader:
        s1_id = row[0].strip()
        m_val = row[1].strip() if len(row) > 1 else ""
        if m_val:
            m_ids = [m.strip() for m in m_val.split(",") if m.strip()]
            sample_gt.append((s1_id, m_ids))
            s1_needed[s1_id] = None
            for mid in m_ids:
                all_matched_ids.add(mid)
            count += 1
            if count >= 25:
                break

# Now scan train_source1 for these S1 IDs
s1_records = {}
with open(os.path.join(dataset_dir, "train", "train_source1.tsv"), "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    h = next(reader)
    for row in reader:
        if row[0].strip() in s1_needed:
            s1_records[row[0].strip()] = row
            if len(s1_records) == len(s1_needed):
                break

# Scan train_source2 and train_source3 for matched IDs
target_s2 = {mid for mid in all_matched_ids if mid.startswith("S2-")}
target_s3 = {mid for mid in all_matched_ids if mid.startswith("S3-")}

s2_records = {}
if target_s2:
    with open(os.path.join(dataset_dir, "train", "train_source2.tsv"), "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        h = next(reader)
        for row in reader:
            if row[0].strip() in target_s2:
                s2_records[row[0].strip()] = row
                if len(s2_records) == len(target_s2):
                    break

s3_records = {}
if target_s3:
    with open(os.path.join(dataset_dir, "train", "train_source3.tsv"), "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        h = next(reader)
        for row in reader:
            if row[0].strip() in target_s3:
                s3_records[row[0].strip()] = row
                if len(s3_records) == len(target_s3):
                    break

print("=== SAMPLE GROUND TRUTH MATCHES (S1 vs S2/S3) ===")
for s1_id, m_ids in sample_gt[:15]:
    s1 = s1_records.get(s1_id, ["?","?","?","?"])
    print(f"\n[S1] ID: {s1_id} | Country: {s1[3]}")
    print(f"     Name:    {s1[1]}")
    print(f"     Address: {s1[2]}")
    for mid in m_ids:
        rec = s2_records.get(mid) or s3_records.get(mid) or ["?","?","?","?"]
        print(f"  --> MATCH [{mid}] Country: {rec[3]}")
        print(f"      Name:    {rec[1]}")
        print(f"      Address: {rec[2]}")
