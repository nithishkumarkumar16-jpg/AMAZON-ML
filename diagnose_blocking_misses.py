import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Force utf-8 output to prevent Windows charmap encoding errors
sys.stdout.reconfigure(encoding="utf-8")

src_dir = Path("code/business_entity_resolution/src").resolve()
sys.path.insert(0, str(src_dir))

from normalization import normalize_business_name, normalize_business_address, normalize_country
from blocking import get_blocking_keys
from predict_v2 import get_v2_blocking_keys

print("=" * 65)
print("DIAGNOSIS OF BLOCKING MISSES ON 1,000 DEVELOPMENT ENTITIES")
print("=" * 65)

# 1. Load the 1,000 dev entities (lines 20,000 to 25,000)
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

# 2. Load ground truth
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
total_true_pairs = len(all_true_cids)
print(f"Loaded {len(dev_s1):,} Dev S1 entities with {total_true_pairs:,} true match pairs.")

# 3. Load true candidates from S2 and S3 directly
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
                if len(true_records) == total_true_pairs: break
    if len(true_records) == total_true_pairs: break

print(f"Loaded {len(true_records):,} true candidate records.")

# 4. Check shared blocking keys between S1 and True S2/S3
v1_shared = 0
v2_shared = 0
missed_pairs = []

for sid, (sid, s1_n, s1_a, s1_c) in dev_s1.items():
    s1_v1_keys = set(get_blocking_keys(s1_n))
    s1_v2_keys = set(get_v2_blocking_keys(s1_n))
    
    for cid in dev_gt[sid]:
        if cid not in true_records: continue
        _, cn, ca, cc = true_records[cid]
        c_v1_keys = set(get_blocking_keys(cn))
        c_v2_keys = set(get_v2_blocking_keys(cn))
        
        has_v1 = bool(s1_v1_keys & c_v1_keys)
        has_v2 = bool(s1_v2_keys & c_v2_keys)
        
        if has_v1: v1_shared += 1
        if has_v2: v2_shared += 1
        else:
            missed_pairs.append((sid, s1_n, s1_a, cid, cn, ca, s1_c))

print("\n--- KEY OVERLAP (BEFORE BUCKET CAP PRUNING) ---")
print(f"V1 Blocking Key Recall: {v1_shared:,} / {len(true_records):,} ({v1_shared/len(true_records)*100:.2f}%)")
print(f"V2 Blocking Key Recall: {v2_shared:,} / {len(true_records):,} ({v2_shared/len(true_records)*100:.2f}%)")
print(f"Missed by V2 Key Generation: {len(missed_pairs):,} pairs ({len(missed_pairs)/len(true_records)*100:.2f}%)")

# 5. Categorize Missed Pairs
categories = Counter()
sample_cases = defaultdict(list)

for sid, s1_n, s1_a, cid, cn, ca, country in missed_pairs:
    s1_toks = set(s1_n.split())
    c_toks = set(cn.split())
    s1_atoks = set(s1_a.split())
    c_atoks = set(ca.split())
    
    common_words = [w for w in (s1_toks & c_toks) if len(w) >= 3]
    addr_overlap = len(s1_atoks & c_atoks)
    
    s1_words = s1_n.split()
    c_words = cn.split()
    w2_match = (len(s1_words) >= 2 and len(c_words) >= 2 and s1_words[1] == c_words[1] and len(s1_words[1]) >= 3)
    pfx3_match = (len(s1_n) >= 3 and len(cn) >= 3 and s1_n[:3] == cn[:3])
    
    # Address Street Number Match + City/ZIP match
    s1_nums = {t for t in s1_atoks if any(c.isdigit() for c in t)}
    c_nums = {t for t in c_atoks if any(c.isdigit() for c in t)}
    street_num_match = bool(s1_nums & c_nums)
    
    cat = "Other / Extreme Name Divergence"
    if common_words:
        cat = f"Shared Non-Lead Token (e.g. '{common_words[0]}')"
    elif street_num_match and addr_overlap >= 3:
        cat = f"Exact Street Address Match (shared num & {addr_overlap} addr tokens)"
    elif w2_match:
        cat = f"Second Word Match ('{s1_words[1]}')"
    elif pfx3_match and len(s1_words) == 1:
        cat = "Typo in Single-Word Name (Prefix-3 match)"
    elif addr_overlap >= 3:
        cat = f"High Address Overlap ({addr_overlap} tokens)"
        
    categories[cat] += 1
    if len(sample_cases[cat]) < 3:
        sample_cases[cat].append((s1_n, cn, s1_a, ca))

print("\n--- ROOT CAUSE ANALYSIS OF MISSED TRUE PAIRS ---")
for cat, count in categories.most_common(10):
    pct = count / len(missed_pairs) * 100
    print(f"\n[{count:,} pairs | {pct:.1f}%] {cat}:")
    for s1_n, cn, s1_a, ca in sample_cases[cat]:
        print(f"   S1:   '{s1_n}' | '{s1_a}'")
        print(f"   Cand: '{cn}' | '{ca}'")
