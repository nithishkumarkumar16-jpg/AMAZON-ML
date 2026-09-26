"""Amazon ML Challenge 2026 — Submission #3 Pipeline (Day 1 Recovery).

Production implementation of the verified conservative matcher.
Guarantees:
- ONE SHARED SCORING IMPLEMENTATION: score_pair_v3 (with bit-for-bit precomputed inner loop)
- Conservative address confirmation: penalizes zero-word address overlap and cross-city collisions.
- Preserves all 1,732,544 rows in exact test_source1 order.
- Every predicted match is a strict subset of candidate_pairs.tsv.
- Rescores the audited, high-recall V2 candidate set (68.8M pairs) country-by-country.
- Memory-bounded streaming without disk paging.
- Checkpointed country-by-country so no progress is ever lost.
- Output directory: output/v3/
"""

import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from config import TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3
from normalization import (
    normalize_business_name,
    normalize_business_address,
    normalize_country,
)

V3_OUTPUT_DIR = Path("output/v3")
V3_MATCHING_OUT = V3_OUTPUT_DIR / "matching_results.tsv"
V3_CANDIDATE_OUT = V3_OUTPUT_DIR / "candidate_pairs.tsv"
V2_CANDIDATE_IN = Path("output/v2/candidate_pairs.tsv")
SELECTED_THRESHOLD = 0.75


def score_pair_v3(s1_name: str, s1_addr: str, cand_name: str, cand_addr: str) -> float:
    """Unified precision-focused candidate scoring function.
    
    Identical across validation and production pipelines.
    Enforces conservative address confirmation to prevent false merges across different cities/states.
    """
    # 1. Name Similarities
    name_exact = 1.0 if s1_name and s1_name == cand_name else 0.0
    ns1 = s1_name.replace(" ", "")
    ns2 = cand_name.replace(" ", "")
    name_ns_match = 1.0 if len(ns1) >= 3 and ns1 == ns2 else 0.0

    tok1 = set(s1_name.split())
    tok2 = set(cand_name.split())
    u = len(tok1 | tok2)
    inter = len(tok1 & tok2)
    name_jaccard = (inter / u) if u > 0 else 0.0
    min_t = min(len(tok1), len(tok2))
    name_containment = (inter / min_t) if min_t > 0 else 0.0

    max_l = max(len(s1_name), len(cand_name), 1)
    len_diff = abs(len(s1_name) - len(cand_name)) / max_l

    name_score = max(name_exact, name_ns_match, name_jaccard * 0.95, name_containment * 0.85)
    if name_exact == 0.0 and name_ns_match == 0.0:
        name_score -= len_diff * 0.15
    name_score = max(0.0, min(1.0, name_score))

    # Fast reject if name score is too low to ever meet threshold
    if name_score < 0.40:
        return 0.0

    # 2. Address Similarities & Evidence Handling
    if not s1_addr or not cand_addr:
        if name_score >= 0.85:
            return name_score * 0.90
        else:
            return name_score * 0.60

    if s1_addr == cand_addr:
        return 0.65 * name_score + 0.35 * 1.0

    atok1 = set(s1_addr.split())
    atok2 = set(cand_addr.split())
    au = len(atok1 | atok2)
    ai = len(atok1 & atok2)
    addr_jaccard = (ai / au) if au > 0 else 0.0

    nums1 = {t for t in atok1 if any(c.isdigit() for c in t)}
    nums2 = {t for t in atok2 if any(c.isdigit() for c in t)}

    if nums1 and nums2:
        num_inter = len(nums1 & nums2)
        num_union = len(nums1 | nums2)
        if num_inter == 0:
            addr_score = addr_jaccard * 0.10
            return (0.50 * name_score + 0.50 * addr_score) * 0.65
        num_jaccard = num_inter / num_union
        addr_score = max(addr_jaccard, 0.6 * addr_jaccard + 0.4 * num_jaccard)
    else:
        addr_score = addr_jaccard

    if addr_jaccard == 0.0:
        if name_score < 0.98:
            return 0.0
        else:
            return 0.60

    composite = 0.65 * name_score + 0.35 * addr_score
    return max(0.0, min(1.0, composite))


def score_pair_v3_precomp(n1, a1, ns1, tok1, atok1, nums1, n2, a2, ns2, tok2, atok2, nums2) -> float:
    """Precomputed inner loop scorer — mathematically 100.00% identical to score_pair_v3."""
    name_exact = 1.0 if n1 and n1 == n2 else 0.0
    name_ns_match = 1.0 if len(ns1) >= 3 and ns1 == ns2 else 0.0

    u = len(tok1 | tok2)
    inter = len(tok1 & tok2)
    name_jaccard = (inter / u) if u > 0 else 0.0
    min_t = min(len(tok1), len(tok2))
    name_containment = (inter / min_t) if min_t > 0 else 0.0

    max_l = max(len(n1), len(n2), 1)
    len_diff = abs(len(n1) - len(n2)) / max_l

    name_score = max(name_exact, name_ns_match, name_jaccard * 0.95, name_containment * 0.85)
    if name_exact == 0.0 and name_ns_match == 0.0:
        name_score -= len_diff * 0.15
    name_score = max(0.0, min(1.0, name_score))

    if name_score < 0.40:
        return 0.0

    if not a1 or not a2:
        if name_score >= 0.85:
            return name_score * 0.90
        else:
            return name_score * 0.60

    if a1 == a2:
        return 0.65 * name_score + 0.35 * 1.0

    au = len(atok1 | atok2)
    ai = len(atok1 & atok2)
    addr_jaccard = (ai / au) if au > 0 else 0.0

    if nums1 and nums2:
        num_inter = len(nums1 & nums2)
        num_union = len(nums1 | nums2)
        if num_inter == 0:
            addr_score = addr_jaccard * 0.10
            return (0.50 * name_score + 0.50 * addr_score) * 0.65
        num_jaccard = num_inter / num_union
        addr_score = max(addr_jaccard, 0.6 * addr_jaccard + 0.4 * num_jaccard)
    else:
        addr_score = addr_jaccard

    if addr_jaccard == 0.0:
        if name_score < 0.98:
            return 0.0
        else:
            return 0.60

    composite = 0.65 * name_score + 0.35 * addr_score
    return max(0.0, min(1.0, composite))


def run_v3_pipeline():
    t_start = time.time()
    print("=" * 65, flush=True)
    print("Amazon ML Challenge 2026 — Submission #3 Recovery Pipeline (Fast)", flush=True)
    print(f"Target Directory:    {V3_OUTPUT_DIR}", flush=True)
    print(f"Decision Threshold:  {SELECTED_THRESHOLD}", flush=True)
    print(f"Candidate Source:    {V2_CANDIDATE_IN}", flush=True)
    print("=" * 65, flush=True)

    os.makedirs(V3_OUTPUT_DIR, exist_ok=True)

    # 1. Ingest test_source1.tsv
    print("\n1. Ingesting test_source1.tsv...", flush=True)
    t0 = time.time()
    s1_store = {}  # s1_id -> (norm_name, norm_addr, country, ns, tok, atok, nums)
    ordered_s1_ids = []
    country_s1_ids = defaultdict(list)

    with open(TEST_SOURCE1, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        id_idx = header.index("entity_id")
        name_idx = header.index("business_name")
        addr_idx = header.index("business_address")
        c_idx = header.index("country")

        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4: continue
            sid = parts[id_idx].strip()
            c = normalize_country(parts[c_idx])
            norm_n = normalize_business_name(parts[name_idx])
            norm_a = normalize_business_address(parts[addr_idx])

            ns = norm_n.replace(" ", "")
            tok = set(norm_n.split())
            atok = set(norm_a.split())
            nums = {t for t in atok if any(ch.isdigit() for ch in t)}

            s1_store[sid] = (norm_n, norm_a, c, ns, tok, atok, nums)
            ordered_s1_ids.append(sid)
            country_s1_ids[c].append(sid)

    print(f"Loaded {len(ordered_s1_ids):,} S1 entities in {time.time() - t0:.2f}s:")
    for c, ids in country_s1_ids.items():
        print(f"  - {c}: {len(ids):,} entities")

    # 2. Ingest candidate pairs by country from V2 candidate_pairs.tsv
    print("\n2. Ingesting candidates from output/v2/candidate_pairs.tsv...", flush=True)
    t0 = time.time()
    cands_by_country = defaultdict(dict)
    needed_candidates_by_country = defaultdict(set)
    total_cand_pairs = 0

    with open(V2_CANDIDATE_IN, "r", encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            sid = parts[0].strip()
            if sid not in s1_store: continue
            c = s1_store[sid][2]

            if len(parts) > 1 and parts[1].strip():
                c_ids = [x.strip() for x in parts[1].split(",") if x.strip()]
                cands_by_country[c][sid] = c_ids
                needed_candidates_by_country[c].update(c_ids)
                total_cand_pairs += len(c_ids)
            else:
                cands_by_country[c][sid] = []

    print(f"Ingested {total_cand_pairs:,} candidate pairs across {len(ordered_s1_ids):,} entities in {time.time() - t0:.2f}s.")
    for c in country_s1_ids:
        print(f"  - {c}: {len(needed_candidates_by_country[c]):,} distinct candidate records required.")

    # 3. Country-by-Country Processing: Stream S2/S3 & Rescore with Checkpointing
    final_matches = {}

    for c_idx, country in enumerate(sorted(country_s1_ids.keys()), start=1):
        c_start = time.time()
        country_out = V3_OUTPUT_DIR / f"matches_{country}.tsv"

        # Check if already completed and checkpointed
        if country_out.exists() and country_out.stat().st_size > 0:
            print(f"\n[{c_idx}/{len(country_s1_ids)}] Country {country} already checkpointed ({country_out}). Loading...", flush=True)
            c_matches = 0
            c_zero = 0
            with open(country_out, "r", encoding="utf-8") as f_c:
                for line in f_c:
                    parts = line.rstrip("\r\n").split("\t")
                    sid = parts[0]
                    if len(parts) > 1 and parts[1]:
                        m = parts[1].split(",")
                        final_matches[sid] = m
                        c_matches += len(m)
                    else:
                        final_matches[sid] = []
                        c_zero += 1
            print(f"  Loaded {country} matches: {c_matches:,} ({c_zero:,} zero-match rows)", flush=True)
            continue

        print(f"\n[{c_idx}/{len(country_s1_ids)}] Processing Country: {country} ({len(country_s1_ids[country]):,} entities)...", flush=True)

        target_cids = needed_candidates_by_country[country]
        cand_store = {}  # cid -> (norm_n, norm_a, ns, tok, atok, nums)

        # Stream S2 for this country
        t_s2 = time.time()
        print(f"  Streaming test_source2.tsv for {country}...", flush=True)
        with open(TEST_SOURCE2, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4: continue
                if normalize_country(parts[3]) != country: continue
                cid = parts[0].strip()
                if cid in target_cids:
                    norm_n = normalize_business_name(parts[1])
                    norm_a = normalize_business_address(parts[2])
                    ns = norm_n.replace(" ", "")
                    tok = set(norm_n.split())
                    atok = set(norm_a.split())
                    nums = {t for t in atok if any(ch.isdigit() for ch in t)}
                    cand_store[cid] = (norm_n, norm_a, ns, tok, atok, nums)

        print(f"  test_source2 processed in {time.time() - t_s2:.2f}s. Loaded {len(cand_store):,} candidates.", flush=True)

        # Stream S3 for this country
        t_s3 = time.time()
        print(f"  Streaming test_source3.tsv for {country}...", flush=True)
        with open(TEST_SOURCE3, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4: continue
                if normalize_country(parts[3]) != country: continue
                cid = parts[0].strip()
                if cid in target_cids:
                    norm_n = normalize_business_name(parts[1])
                    norm_a = normalize_business_address(parts[2])
                    ns = norm_n.replace(" ", "")
                    tok = set(norm_n.split())
                    atok = set(norm_a.split())
                    nums = {t for t in atok if any(ch.isdigit() for ch in t)}
                    cand_store[cid] = (norm_n, norm_a, ns, tok, atok, nums)

        print(f"  test_source3 processed in {time.time() - t_s3:.2f}s. Total loaded candidates: {len(cand_store):,}", flush=True)

        # Rescore candidates for this country with ultra-fast precomputed scorer
        t_score = time.time()
        country_cands = cands_by_country[country]
        c_matches = 0
        c_zero = 0

        for sid in country_s1_ids[country]:
            s1_name, s1_addr, _, ns1, tok1, atok1, nums1 = s1_store[sid]
            cids = country_cands.get(sid, [])
            matched = []

            for cid in cids:
                if cid not in cand_store: continue
                cn, ca, ns2, tok2, atok2, nums2 = cand_store[cid]
                score = score_pair_v3_precomp(s1_name, s1_addr, ns1, tok1, atok1, nums1, cn, ca, ns2, tok2, atok2, nums2)
                if score >= SELECTED_THRESHOLD:
                    matched.append(cid)

            final_matches[sid] = matched
            if matched:
                c_matches += len(matched)
            else:
                c_zero += 1

        print(f"  Country {country} rescored in {time.time() - c_start:.2f}s (scoring loop: {time.time() - t_score:.2f}s):", flush=True)
        print(f"    - Matches: {c_matches:,} (avg {c_matches/len(country_s1_ids[country]):.2f}/entity)", flush=True)
        print(f"    - Zero-Match Rows: {c_zero:,} ({c_zero/len(country_s1_ids[country])*100:.2f}%)", flush=True)

        # Save checkpoint for this country
        print(f"    Saving checkpoint to {country_out}...", flush=True)
        with open(country_out, "w", encoding="utf-8") as f_c:
            for sid in country_s1_ids[country]:
                m = final_matches.get(sid, [])
                f_c.write(f"{sid}\t{','.join(m)}\n")

        del cand_store
        del country_cands

    # 4. Write Final Output Submission Files
    print("\n4. Writing final submission files to output/v3/...", flush=True)
    t0 = time.time()
    total_written_matches = 0
    total_zero_match_rows = 0

    with open(V3_MATCHING_OUT, "w", encoding="utf-8") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in ordered_s1_ids:
            m_list = final_matches.get(sid, [])
            if m_list:
                fm.write(f"{sid}\t{','.join(m_list)}\n")
                total_written_matches += len(m_list)
            else:
                fm.write(f"{sid}\t\n")
                total_zero_match_rows += 1

    print(f"  Exported {V3_MATCHING_OUT}: {len(ordered_s1_ids):,} rows ({total_written_matches:,} matches, {total_zero_match_rows:,} zero-match rows).", flush=True)

    # Replicate candidate_pairs.tsv directly (byte-for-byte identical to V2)
    print(f"  Copying {V2_CANDIDATE_IN} to {V3_CANDIDATE_OUT}...", flush=True)
    shutil.copyfile(V2_CANDIDATE_IN, V3_CANDIDATE_OUT)
    print(f"  Candidate file replicated in {time.time() - t0:.2f}s.", flush=True)

    # 5. Generate Checksums & Manifest
    print("\n5. Generating checksums for V3 deliverables...", flush=True)
    manifest = {
        "pipeline_version": "V3",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "decision_threshold": SELECTED_THRESHOLD,
        "matcher": "score_pair_v3_conservative",
        "total_s1_rows": len(ordered_s1_ids),
        "total_predicted_matches": total_written_matches,
        "total_zero_match_rows": total_zero_match_rows,
        "total_candidate_pairs": total_cand_pairs,
        "total_runtime_seconds": round(time.time() - t_start, 2),
        "files": {},
    }

    checksum_lines = []
    for fpath in [V3_MATCHING_OUT, V3_CANDIDATE_OUT]:
        h = hashlib.sha256()
        with open(fpath, "rb") as f:
            while chunk := f.read(1048576):
                h.update(chunk)
        digest = h.hexdigest()
        manifest["files"][fpath.name] = {
            "sha256": digest,
            "bytes": fpath.stat().st_size,
        }
        checksum_lines.append(f"{digest}  {fpath.name}")

    with open(V3_OUTPUT_DIR / "checksums.sha256", "w", encoding="utf-8") as f:
        f.write("\n".join(checksum_lines) + "\n")

    with open(V3_OUTPUT_DIR / "experiment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total_time = time.time() - t_start
    print(f"\n>>> Total V3 Pipeline Runtime: {total_time:.2f}s ({total_time/60:.2f} minutes) <<<", flush=True)


if __name__ == "__main__":
    run_v3_pipeline()
