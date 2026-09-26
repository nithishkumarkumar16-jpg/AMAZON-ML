"""End-to-end inference script for Day 1 Submission #1.

Generates:
- output/matching_results.tsv
- output/candidate_pairs.tsv

Adheres to all competition rules:
- sep='\t' formatting
- Strict country partitioning (open set including France)
- Memory-safe country-by-country streaming
- Strict subset guarantee: matches are always a subset of candidates
- Exact match to test_source1.tsv row order and count (1,732,544 rows)
"""

import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Add package source to path
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from config import (
    CANDIDATE_COLUMNS,
    CANDIDATE_PAIRS_PATH,
    DELIMITER,
    MATCHING_RESULTS_PATH,
    SOURCE_COLUMNS,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
)
from normalization import (
    normalize_business_name,
    normalize_business_address,
    normalize_country,
)
from blocking import BlockingIndex, get_blocking_keys
from matcher import BaselineMatcher


def generate_test_predictions(
    test_s1_path: Path = TEST_SOURCE1,
    test_s2_path: Path = TEST_SOURCE2,
    test_s3_path: Path = TEST_SOURCE3,
    matching_out_path: Path = MATCHING_RESULTS_PATH,
    candidate_out_path: Path = CANDIDATE_PAIRS_PATH,
    threshold: float = 0.75,
    max_bucket_size: int = 150,
):
    """Run full test inference and produce valid submission files."""
    start_time = time.time()
    print("=" * 60, flush=True)
    print("Amazon ML Challenge 2026 — Day 1 Baseline Inference", flush=True)
    print(f"Selected Decision Threshold: {threshold:.2f}", flush=True)
    print("=" * 60, flush=True)

    # --------------------------------------------------------------------------
    # 1. Load & Partition Test Source 1
    # --------------------------------------------------------------------------
    t0 = time.time()
    print(f"\n1. Ingesting test_source1.tsv...", flush=True)

    ordered_s1_ids = []
    # country -> dict of s1_id -> (norm_name, norm_addr)
    s1_by_country = defaultdict(dict)
    # country -> set of target blocking keys
    target_keys_by_country = defaultdict(set)

    with open(test_s1_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        col_idx = {col: i for i, col in enumerate(header)}
        
        for line_num, line in enumerate(f, start=2):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                continue
            
            s1_id = parts[col_idx["entity_id"]].strip()
            raw_name = parts[col_idx["business_name"]]
            raw_addr = parts[col_idx["business_address"]]
            raw_country = parts[col_idx["country"]]

            country = normalize_country(raw_country)
            norm_name = normalize_business_name(raw_name)
            norm_addr = normalize_business_address(raw_addr)

            ordered_s1_ids.append(s1_id)
            s1_by_country[country][s1_id] = (norm_name, norm_addr)

            keys = get_blocking_keys(norm_name)
            target_keys_by_country[country].update(keys)

    n_total_s1 = len(ordered_s1_ids)
    print(f"Loaded {n_total_s1:,} S1 entities across {len(s1_by_country)} countries in {time.time() - t0:.2f}s:")
    for country, entities in s1_by_country.items():
        print(f"  - {country}: {len(entities):,} entities | {len(target_keys_by_country[country]):,} target keys")

    # Storage for final results: s1_id -> comma-separated string
    results_matches = {}
    results_candidates = {}

    matcher = BaselineMatcher(threshold=threshold)

    # --------------------------------------------------------------------------
    # 2. Process Country-by-Country (Memory Conscious)
    # --------------------------------------------------------------------------
    # Open-set iteration over every country present in test_source1
    countries = sorted(s1_by_country.keys())

    for c_idx, country in enumerate(countries, start=1):
        c_start = time.time()
        country_s1 = s1_by_country[country]
        country_target_keys = target_keys_by_country[country]

        print(f"\n[{c_idx}/{len(countries)}] Processing Country: {country} ({len(country_s1):,} S1 entities)...", flush=True)

        index = BlockingIndex(max_bucket_size=max_bucket_size)
        cand_store = {}  # cid -> (norm_name, norm_addr)

        # Stream Test Source 2 for this country
        t_s2 = time.time()
        print(f"  Streaming test_source2.tsv for {country}...", flush=True)
        with open(test_s2_path, "r", encoding="utf-8") as f:
            f.readline()  # Skip header
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4:
                    continue
                if normalize_country(parts[3]) != country:
                    continue
                
                cname = parts[1]
                norm_n = normalize_business_name(cname)
                keys = get_blocking_keys(norm_n)
                matching = [k for k in keys if k in country_target_keys]
                if matching:
                    cid = parts[0].strip()
                    norm_a = normalize_business_address(parts[2])
                    index.add_record(cid, matching)
                    cand_store[cid] = (norm_n, norm_a)

        print(f"  test_source2 processed in {time.time() - t_s2:.2f}s. Stored candidates so far: {len(cand_store):,}", flush=True)

        # Stream Test Source 3 for this country
        t_s3 = time.time()
        print(f"  Streaming test_source3.tsv for {country}...", flush=True)
        with open(test_s3_path, "r", encoding="utf-8") as f:
            f.readline()  # Skip header
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4:
                    continue
                if normalize_country(parts[3]) != country:
                    continue
                
                cname = parts[1]
                norm_n = normalize_business_name(cname)
                keys = get_blocking_keys(norm_n)
                matching = [k for k in keys if k in country_target_keys]
                if matching:
                    cid = parts[0].strip()
                    norm_a = normalize_business_address(parts[2])
                    index.add_record(cid, matching)
                    cand_store[cid] = (norm_n, norm_a)

        print(f"  test_source3 processed in {time.time() - t_s3:.2f}s. Total stored candidates: {len(cand_store):,}", flush=True)

        # Prune high-frequency buckets
        index.filter_oversized_buckets()

        # Match S1 entities
        t_match = time.time()
        print(f"  Scoring candidates for {len(country_s1):,} S1 entities...", flush=True)
        c_matches_count = 0
        c_singletons_count = 0

        for s1_id, (s1_name, s1_addr) in country_s1.items():
            cand_ids = index.get_candidates_for_name(s1_name)
            cand_records = [
                (cid, cand_store[cid][0], cand_store[cid][1])
                for cid in cand_ids
                if cid in cand_store
            ]

            eval_cand_ids, matched_ids = matcher.match_candidates(
                s1_name, s1_addr, cand_records
            )

            # Store formatted strings
            results_candidates[s1_id] = ",".join(eval_cand_ids)
            results_matches[s1_id] = ",".join(matched_ids)

            if matched_ids:
                c_matches_count += len(matched_ids)
            else:
                c_singletons_count += 1

        print(f"  Country {country} completed in {time.time() - c_start:.2f}s:")
        print(f"    - Predicted Matches: {c_matches_count:,}")
        print(f"    - Predicted Singletons: {c_singletons_count:,} ({c_singletons_count/len(country_s1)*100:.2f}%)")

        # Explicitly clean memory before next country
        del index
        del cand_store

    # --------------------------------------------------------------------------
    # 3. Export Output TSV Files in Exact Original Order
    # --------------------------------------------------------------------------
    t_write = time.time()
    print(f"\n3. Writing output submission files...", flush=True)
    os.makedirs(matching_out_path.parent, exist_ok=True)

    # 3a. matching_results.tsv
    total_written_matches = 0
    total_singletons = 0
    with open(matching_out_path, "w", encoding="utf-8", newline="\n") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in ordered_s1_ids:
            m_str = results_matches.get(s1_id, "")
            f_match.write(f"{s1_id}\t{m_str}\n")
            if m_str:
                total_written_matches += len(m_str.split(","))
            else:
                total_singletons += 1

    print(f"  Exported {matching_out_path}: {n_total_s1:,} rows ({total_written_matches:,} matches, {total_singletons:,} singletons).")

    # 3b. candidate_pairs.tsv
    total_written_cands = 0
    with open(candidate_out_path, "w", encoding="utf-8", newline="\n") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in ordered_s1_ids:
            c_str = results_candidates.get(s1_id, "")
            f_cand.write(f"{s1_id}\t{c_str}\n")
            if c_str:
                total_written_cands += len(c_str.split(","))

    print(f"  Exported {candidate_out_path}: {n_total_s1:,} rows ({total_written_cands:,} total candidate pairs).")
    print(f"\n>>> Total Pipeline Runtime: {time.time() - start_time:.2f}s <<<", flush=True)


if __name__ == "__main__":
    generate_test_predictions()
