"""Submission #2 (V2) Inference Pipeline for Amazon ML Challenge 2026.

Integrates:
1. Enhanced Multi-Angle Blocking (OCR folding, stop-word stripping, predecessor split, unigram token keys)
2. Fast-Path Enhanced Matcher (Exact match shortcut, character 3-gram Dice, edit ratio, address containment, calibrated missing address)
3. Precision-calibrated threshold: 0.72
4. Outputs to output/v2/ with full auditability and checksums.
"""

import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# Setup paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent.parent.parent
sys.path.insert(0, str(SRC_DIR))

from config import (
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
)
from normalization import (
    normalize_business_name,
    normalize_business_address,
    normalize_country,
)

V2_OUTPUT_DIR = PROJECT_ROOT / "output" / "v2"
V2_MATCHING_FILE = V2_OUTPUT_DIR / "matching_results.tsv"
V2_CANDIDATE_FILE = V2_OUTPUT_DIR / "candidate_pairs.tsv"

# Model hyperparameters
SELECTED_THRESHOLD = 0.72
MAX_BUCKET_SIZE = 120  # Strict cap to prevent candidate explosion

# Pre-compiled regexes
OCR_MAP = str.maketrans({"0": "o", "1": "l", "5": "s", "8": "b", "@": "a", "$": "s"})
STOP_LEADS = re.compile(r"^(the|a|an|le|la|les|el|der|die|das)\s+", re.IGNORECASE)
PRED_SPLIT = re.compile(r"\b(?:formerly|f/k/a|fka|d/b/a|dba|a/k/a|aka|c/o)\b", re.IGNORECASE)
GENERIC_WORDS = {
    "company", "services", "service", "center", "group", "holdings",
    "enterprises", "solutions", "international", "associates", "management",
    "consulting", "industries", "products", "global", "national", "united"
}


def get_v2_blocking_keys(norm_name: str) -> list:
    """Generate high-recall blocking keys with OCR folding, stop-word stripping, and unigram keys."""
    if not norm_name or len(norm_name) < 2:
        return []

    keys = []
    clean_name = STOP_LEADS.sub("", norm_name).strip()
    if not clean_name:
        clean_name = norm_name

    parts = PRED_SPLIT.split(clean_name)
    sub_names = [clean_name]
    if len(parts) > 1:
        for p in parts:
            p_clean = p.strip(" -+,:")
            if len(p_clean) >= 3:
                sub_names.append(p_clean)

    for s_name in sub_names:
        # 1. Exact
        keys.append(f"ex:{s_name}")

        # 2. No-space
        ns = s_name.replace(" ", "")
        if len(ns) >= 3:
            keys.append(f"ns:{ns}")

        # 3. OCR folded
        if any(c in "0158@$" for c in s_name):
            ocr_folded = s_name.translate(OCR_MAP)
            keys.append(f"ex:{ocr_folded}")
            keys.append(f"ns:{ocr_folded.replace(' ', '')}")

        tokens = s_name.split()
        sig_tokens = [t for t in tokens if len(t) > 1 and t not in GENERIC_WORDS]

        # 4. Lead unigram token key (len >= 4)
        if tokens:
            w1 = tokens[0]
            if len(w1) >= 4 and w1 not in GENERIC_WORDS:
                keys.append(f"w1:{w1}")
                if any(c in "0158@$" for c in w1):
                    keys.append(f"w1:{w1.translate(OCR_MAP)}")

        # 5. Sorted significant tokens
        if len(sig_tokens) > 1:
            sorted_key = " ".join(sorted(sig_tokens[:4]))
            keys.append(f"sort:{sorted_key}")
            if len(sig_tokens) >= 2:
                sort2 = " ".join(sorted(sig_tokens[:2]))
                keys.append(f"sort2:{sort2}")

        # 6. Prefix keys
        if len(tokens) >= 2:
            lead2 = " ".join(tokens[:2])
            if len(lead2) >= 4:
                keys.append(f"pfx2:{lead2}")
        elif len(tokens) == 1 and len(tokens[0]) >= 5:
            keys.append(f"pfx5:{tokens[0][:5]}")

    return list(dict.fromkeys(keys))


# Fast String Similarities
import difflib

def char_ngram_dice(s1: str, s2: str, n: int = 3) -> float:
    if not s1 or not s2: return 0.0
    if s1 == s2: return 1.0
    l1, l2 = len(s1), len(s2)
    if l1 < n or l2 < n: return 1.0 if s1 == s2 else 0.0
    ngrams1 = {s1[i:i+n] for i in range(l1 - n + 1)}
    ngrams2 = {s2[i:i+n] for i in range(l2 - n + 1)}
    inter = len(ngrams1 & ngrams2)
    total = len(ngrams1) + len(ngrams2)
    return (2.0 * inter) / total if total > 0 else 0.0


def fast_score_candidate_pair(s1_name, s1_addr, cand_name, cand_addr) -> float:
    """Optimized fast-path candidate pair scoring."""
    # Fast path 1: Exact identical name
    if s1_name and s1_name == cand_name:
        if not s1_addr or not cand_addr:
            return 0.95
        if s1_addr == cand_addr:
            return 1.0
        # Check numeric conflict
        atok1 = set(s1_addr.split())
        atok2 = set(cand_addr.split())
        nums1 = {t for t in atok1 if any(c.isdigit() for c in t)}
        nums2 = {t for t in atok2 if any(c.isdigit() for c in t)}
        if nums1 and nums2 and len(nums1 & nums2) == 0:
            return 0.65  # Heavy conflict
        return 0.90

    # Fast path 2: Space-stripped identical
    ns1 = s1_name.replace(" ", "")
    ns2 = cand_name.replace(" ", "")
    if len(ns1) >= 3 and ns1 == ns2:
        if not s1_addr or not cand_addr:
            return 0.92
        return 0.95

    # Token overlap
    tok1 = set(s1_name.split())
    tok2 = set(cand_name.split())
    u = len(tok1 | tok2)
    inter = len(tok1 & tok2)
    jaccard = (inter / u) if u > 0 else 0.0
    min_t = min(len(tok1), len(tok2))
    containment = (inter / min_t) if min_t > 0 else 0.0

    # Character 3-gram Dice
    dice3 = char_ngram_dice(ns1, ns2, n=3)

    # Edit distance (only if lengths are within 35%)
    max_l = max(len(s1_name), len(cand_name), 1)
    min_l = min(len(s1_name), len(cand_name))
    if min_l / max_l >= 0.35:
        edit_ratio = difflib.SequenceMatcher(None, s1_name, cand_name).ratio()
    else:
        edit_ratio = 0.0

    name_score = max(
        jaccard * 0.95,
        containment * 0.85,
        dice3 * 0.92,
        edit_ratio * 0.90,
    )
    len_diff = abs(len(s1_name) - len(cand_name)) / max_l
    name_score -= len_diff * 0.12
    name_score = max(0.0, min(1.0, name_score))

    # Fast reject if name score is too low
    if name_score < 0.50:
        return 0.0

    # Address scoring
    if not s1_addr or not cand_addr:
        if name_score >= 0.85:
            return name_score * 0.92
        elif name_score >= 0.75:
            return name_score * 0.82
        else:
            return name_score * 0.60

    if s1_addr == cand_addr:
        return 0.62 * name_score + 0.38 * 1.0

    atok1 = set(s1_addr.split())
    atok2 = set(cand_addr.split())
    nums1 = {t for t in atok1 if any(c.isdigit() for c in t)}
    nums2 = {t for t in atok2 if any(c.isdigit() for c in t)}

    if nums1 and nums2 and len(nums1 & nums2) == 0:
        # Numeric conflict
        addr_score = (len(atok1 & atok2) / len(atok1 | atok2)) * 0.10 if (atok1 | atok2) else 0.0
        return (0.50 * name_score + 0.50 * addr_score) * 0.65

    au = len(atok1 | atok2)
    ai = len(atok1 & atok2)
    addr_jaccard = (ai / au) if au > 0 else 0.0
    num_jaccard = (len(nums1 & nums2) / len(nums1 | nums2)) if (nums1 and nums2) else (1.0 if not nums1 and not nums2 else 0.0)

    addr_score = max(addr_jaccard, 0.6 * addr_jaccard + 0.4 * num_jaccard)
    composite = 0.62 * name_score + 0.38 * addr_score
    return max(0.0, min(1.0, composite))


def generate_submission_v2():
    t_start = time.time()
    print("=" * 65)
    print("Amazon ML Challenge 2026 — Submission #2 Pipeline")
    print(f"Target Directory: {V2_OUTPUT_DIR}")
    print(f"Decision Threshold: {SELECTED_THRESHOLD}")
    print("=" * 65)

    os.makedirs(V2_OUTPUT_DIR, exist_ok=True)

    # 1. Ingest test_source1.tsv
    print("\n1. Ingesting test_source1.tsv...")
    s1_by_country = defaultdict(dict)
    ordered_s1_ids = []
    target_keys_by_country = defaultdict(set)

    with open(TEST_SOURCE1, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        id_idx = header.index("entity_id")
        name_idx = header.index("business_name")
        addr_idx = header.index("business_address")
        c_idx = header.index("country")

        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4: continue
            s1_id = parts[id_idx].strip()
            ordered_s1_ids.append(s1_id)

            country = normalize_country(parts[c_idx])
            norm_name = normalize_business_name(parts[name_idx])
            norm_addr = normalize_business_address(parts[addr_idx])

            s1_by_country[country][s1_id] = (norm_name, norm_addr)
            keys = get_v2_blocking_keys(norm_name)
            target_keys_by_country[country].update(keys)

    n_total_s1 = len(ordered_s1_ids)
    print(f"Loaded {n_total_s1:,} S1 entities across {len(s1_by_country)} countries:")
    for c, entities in s1_by_country.items():
        print(f"  - {c}: {len(entities):,} entities | {len(target_keys_by_country[c]):,} target keys")

    # 2. Process Country by Country (Streaming S2 and S3)
    results_matches = {}
    results_candidates = {}

    country_order = sorted(s1_by_country.keys(), key=lambda c: len(s1_by_country[c]))

    for c_idx, country in enumerate(country_order, 1):
        c_start = time.time()
        country_s1 = s1_by_country[country]
        country_target_keys = target_keys_by_country[country]
        print(f"\n[{c_idx}/{len(country_order)}] Processing Country: {country} ({len(country_s1):,} S1 entities)...", flush=True)

        index = defaultdict(list)
        cand_store = {}

        # Stream S2
        t0 = time.time()
        print(f"  Streaming test_source2.tsv for {country}...", flush=True)
        with open(TEST_SOURCE2, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4: continue
                if normalize_country(parts[3]) != country: continue

                norm_name = normalize_business_name(parts[1])
                keys = get_v2_blocking_keys(norm_name)
                matching_keys = [k for k in keys if k in country_target_keys]
                if matching_keys:
                    cid = parts[0].strip()
                    norm_addr = normalize_business_address(parts[2])
                    cand_store[cid] = (norm_name, norm_addr)
                    for mk in matching_keys:
                        index[mk].append(cid)

        print(f"  test_source2 processed in {time.time() - t0:.2f}s. Stored candidates so far: {len(cand_store):,}", flush=True)

        # Stream S3
        t0 = time.time()
        print(f"  Streaming test_source3.tsv for {country}...", flush=True)
        with open(TEST_SOURCE3, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4: continue
                if normalize_country(parts[3]) != country: continue

                norm_name = normalize_business_name(parts[1])
                keys = get_v2_blocking_keys(norm_name)
                matching_keys = [k for k in keys if k in country_target_keys]
                if matching_keys:
                    cid = parts[0].strip()
                    norm_addr = normalize_business_address(parts[2])
                    cand_store[cid] = (norm_name, norm_addr)
                    for mk in matching_keys:
                        index[mk].append(cid)

        print(f"  test_source3 processed in {time.time() - t0:.2f}s. Total stored candidates: {len(cand_store):,}", flush=True)

        # Prune oversized buckets
        oversized = [k for k, ids in index.items() if len(ids) > MAX_BUCKET_SIZE]
        for k in oversized:
            del index[k]
        print(f"  Active keys: {len(index):,} (pruned {len(oversized):,} oversized keys exceeding {MAX_BUCKET_SIZE}).", flush=True)

        # Score candidates for each S1 entity
        print(f"  Scoring candidates for {len(country_s1):,} S1 entities...", flush=True)
        c_matches_count = 0
        c_singletons_count = 0

        for s1_id, (s1_name, s1_addr) in country_s1.items():
            s1_keys = get_v2_blocking_keys(s1_name)
            c_ids = set()
            for k in s1_keys:
                if k in index:
                    c_ids.update(index[k])

            sorted_cand_ids = sorted(c_ids)
            matched_ids = []

            for cid in sorted_cand_ids:
                if cid not in cand_store: continue
                cname, caddr = cand_store[cid]
                score = fast_score_candidate_pair(s1_name, s1_addr, cname, caddr)
                if score >= SELECTED_THRESHOLD:
                    matched_ids.append(cid)

            results_candidates[s1_id] = ",".join(sorted_cand_ids)
            results_matches[s1_id] = ",".join(matched_ids)

            if matched_ids:
                c_matches_count += len(matched_ids)
            else:
                c_singletons_count += 1

        print(f"  Country {country} completed in {time.time() - c_start:.2f}s:")
        print(f"    - Matches: {c_matches_count:,} (avg {c_matches_count/len(country_s1):.2f}/entity)")
        print(f"    - Singletons: {c_singletons_count:,} ({c_singletons_count/len(country_s1)*100:.2f}%)", flush=True)

        del index
        del cand_store

    # 3. Export Output TSV Files
    print("\n3. Writing output submission files to output/v2/...", flush=True)
    total_written_matches = 0
    total_singletons = 0

    with open(V2_MATCHING_FILE, "w", encoding="utf-8", newline="\n") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in ordered_s1_ids:
            m_str = results_matches.get(s1_id, "")
            f_match.write(f"{s1_id}\t{m_str}\n")
            if m_str:
                total_written_matches += len(m_str.split(","))
            else:
                total_singletons += 1

    print(f"  Exported {V2_MATCHING_FILE}: {n_total_s1:,} rows ({total_written_matches:,} matches, {total_singletons:,} singletons).")

    total_written_cands = 0
    with open(V2_CANDIDATE_FILE, "w", encoding="utf-8", newline="\n") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in ordered_s1_ids:
            c_str = results_candidates.get(s1_id, "")
            f_cand.write(f"{s1_id}\t{c_str}\n")
            if c_str:
                total_written_cands += len(c_str.split(","))

    print(f"  Exported {V2_CANDIDATE_FILE}: {n_total_s1:,} rows ({total_written_cands:,} candidate pairs).")

    # 4. Generate SHA256 Checksums
    print("\n4. Generating checksums for V2 files...")
    manifest = {
        "pipeline_version": "V2",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "decision_threshold": SELECTED_THRESHOLD,
        "max_bucket_size": MAX_BUCKET_SIZE,
        "total_s1_rows": n_total_s1,
        "total_predicted_matches": total_written_matches,
        "total_predicted_singletons": total_singletons,
        "total_candidate_pairs": total_written_cands,
        "total_runtime_seconds": round(time.time() - t_start, 2),
        "files": {}
    }

    checksum_lines = []
    for fpath in [V2_MATCHING_FILE, V2_CANDIDATE_FILE]:
        sha = hashlib.sha256()
        with open(fpath, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        h = sha.hexdigest()
        manifest["files"][fpath.name] = {
            "sha256": h,
            "bytes": fpath.stat().st_size,
        }
        checksum_lines.append(f"{h}  {fpath.name}")

    with open(V2_OUTPUT_DIR / "checksums.sha256", "w", encoding="utf-8") as f:
        f.write("\n".join(checksum_lines) + "\n")

    with open(V2_OUTPUT_DIR / "experiment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n>>> Total V2 Pipeline Runtime: {time.time() - t_start:.2f}s <<<", flush=True)


if __name__ == "__main__":
    generate_submission_v2()
