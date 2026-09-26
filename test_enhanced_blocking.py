"""Test enhanced blocking strategy on 5,000 validation split.

Evaluates recall gain, candidate volume, and new oracle ceiling.
"""

import re
import sys
import time
import pickle
from collections import defaultdict
from pathlib import Path

src_dir = Path(__file__).resolve().parent / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(src_dir))

from config import TRAIN_SOURCE2, TRAIN_SOURCE3
from normalization import normalize_business_name, normalize_business_address
from evaluation import compute_f05_score

CACHE_FILE = Path("output/val_cache_5k.pkl")
with open(CACHE_FILE, "rb") as f:
    s1_records, gt_records, old_cands = pickle.load(f)

# OCR mapping table
OCR_MAP = str.maketrans({"0": "o", "1": "l", "5": "s", "8": "b", "@": "a", "$": "s"})

STOP_LEADS = re.compile(r"^(the|a|an|le|la|les|el|der|die|das)\s+", re.IGNORECASE)
PRED_SPLIT = re.compile(r"\b(?:formerly|f/k/a|fka|d/b/a|dba|a/k/a|aka|c/o)\b", re.IGNORECASE)
GENERIC_WORDS = {
    "company", "services", "service", "center", "group", "holdings",
    "enterprises", "solutions", "international", "associates", "management",
    "consulting", "industries", "products", "global", "national", "united"
}


def get_enhanced_blocking_keys(norm_name: str) -> list:
    """Generate rich, high-recall blocking keys with OCR folding, stop-word stripping, and token keys."""
    if not norm_name or len(norm_name) < 2:
        return []

    keys = []
    
    # 1. Base cleaned name
    clean_name = STOP_LEADS.sub("", norm_name).strip()
    if not clean_name:
        clean_name = norm_name

    # Check for predecessor / successor split (e.g. "X formerly Y")
    parts = PRED_SPLIT.split(clean_name)
    sub_names = [clean_name]
    if len(parts) > 1:
        for p in parts:
            p_clean = p.strip(" -+,:")
            if len(p_clean) >= 3:
                sub_names.append(p_clean)

    for s_name in sub_names:
        # A. Exact normalized
        keys.append(f"ex:{s_name}")

        # B. No space
        ns = s_name.replace(" ", "")
        if len(ns) >= 3:
            keys.append(f"ns:{ns}")

        # C. OCR folded (replaces 0->o, 1->l, 5->s, etc.)
        if any(c in "0158@$" for c in s_name):
            ocr_folded = s_name.translate(OCR_MAP)
            keys.append(f"ex:{ocr_folded}")
            keys.append(f"ns:{ocr_folded.replace(' ', '')}")

        tokens = s_name.split()
        sig_tokens = [t for t in tokens if len(t) > 1 and t not in GENERIC_WORDS]

        # D. First word key (if len >= 4 and not generic)
        if tokens:
            w1 = tokens[0]
            if len(w1) >= 4 and w1 not in GENERIC_WORDS:
                keys.append(f"w1:{w1}")
                if any(c in "0158@$" for c in w1):
                    keys.append(f"w1:{w1.translate(OCR_MAP)}")

        # E. Sorted significant tokens
        if len(sig_tokens) > 1:
            sorted_key = " ".join(sorted(sig_tokens[:4]))
            keys.append(f"sort:{sorted_key}")
            # Also first 2 sorted tokens
            if len(sig_tokens) >= 2:
                sort2 = " ".join(sorted(sig_tokens[:2]))
                keys.append(f"sort2:{sort2}")

        # F. Lead prefix keys
        if len(tokens) >= 2:
            lead2 = " ".join(tokens[:2])
            if len(lead2) >= 4:
                keys.append(f"pfx2:{lead2}")
        elif len(tokens) == 1 and len(tokens[0]) >= 5:
            keys.append(f"pfx5:{tokens[0][:5]}")

    return list(dict.fromkeys(keys))


if __name__ == "__main__":
    # Collect target keys for validation S1
    target_keys = {"US": set(), "India": set()}
    for s1_id, (_, s1_name, _, country) in s1_records.items():
        if country in target_keys:
            target_keys[country].update(get_enhanced_blocking_keys(s1_name))

    print(f"Target keys generated: US={len(target_keys['US']):,}, India={len(target_keys['India']):,}")

# Build Index from S2 & S3
indexes = {
    "US": defaultdict(list),
    "India": defaultdict(list)
}
cand_store = {}

MAX_BUCKET = 120  # Keep bucket sizes tight to prevent candidate explosion

for s_file, label in [(TRAIN_SOURCE2, "S2"), (TRAIN_SOURCE3, "S3")]:
    t0 = time.time()
    print(f"Streaming {label} with enhanced keys...", flush=True)
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
            keys = get_enhanced_blocking_keys(norm_n)
            matching_keys = [k for k in keys if k in tkeys]
            if matching_keys:
                cid = parts[0]
                norm_a = normalize_business_address(parts[2])
                for mk in matching_keys:
                    indexes[c][mk].append(cid)
                cand_store[cid] = (norm_n, norm_a)
    print(f"{label} done in {time.time() - t0:.2f}s. Stored candidates so far: {len(cand_store):,}")

# Prune oversized buckets
for c in ["US", "India"]:
    oversized = [k for k, v in indexes[c].items() if len(v) > MAX_BUCKET]
    for k in oversized:
        del indexes[c][k]
    print(f"Country {c}: retained {len(indexes[c]):,} active keys (pruned {len(oversized):,} oversized).")

# Retrieve and evaluate candidate recall
total_true_matches = sum(len(m) for m in gt_records.values())
retrieved_true_matches = 0
total_candidates = 0
cand_counts = []
oracle_preds = {}

for s1_id, (_, s1_name, _, country) in s1_records.items():
    s1_keys = get_enhanced_blocking_keys(s1_name)
    c_ids = set()
    idx = indexes.get(country, {})
    for k in s1_keys:
        if k in idx:
            c_ids.update(idx[k])

    cand_counts.append(len(c_ids))
    total_candidates += len(c_ids)
    true_set = gt_records.get(s1_id, set())

    found = len(c_ids & true_set)
    retrieved_true_matches += found
    oracle_preds[s1_id] = c_ids & true_set

cand_counts_sorted = sorted(cand_counts)
n = len(cand_counts)
oracle_f05 = compute_f05_score(gt_records, oracle_preds)

print("\n" + "=" * 70)
print("ENHANCED BLOCKING EVALUATION RESULTS")
print("=" * 70)
print(f"V1 Candidate Recall:       10,716 / 17,361 (61.72%)")
print(f"V2 Candidate Recall:       {retrieved_true_matches:,} / {total_true_matches:,} ({retrieved_true_matches/total_true_matches*100:.2f}%)")
print(f"Recall Improvement:        +{retrieved_true_matches - 10716:,} true matches (+{(retrieved_true_matches/total_true_matches*100) - 61.72:+.2f}%)")
print(f"V1 Avg Candidates/Entity:  26.97")
print(f"V2 Avg Candidates/Entity:  {total_candidates / n:.2f} (Median={cand_counts_sorted[n//2]}, P95={cand_counts_sorted[int(n*0.95)]}, Max={cand_counts_sorted[-1]})")
print(f"V1 Oracle Ceiling F0.5:    0.7830")
print(f"V2 Oracle Ceiling F0.5:    {oracle_f05:.4f} (+{oracle_f05 - 0.7830:+.4f})")
print("=" * 70)

# Save cache for matching experiments
val_cache_v2 = (s1_records, gt_records, indexes, cand_store)
with open("output/val_cache_v2.pkl", "wb") as f:
    pickle.dump(val_cache_v2, f)
print("Saved output/val_cache_v2.pkl for fast matcher development.")
