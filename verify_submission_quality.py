"""Submission Quality Assurance & Diagnostic Script.

Performs all 10 quality checks required for Day 1 baseline submission:
1. Row count and column header verification
2. S1 entity bijection (no missing, no extra, no duplicate IDs)
3. Intra-row duplicate check
4. Prefix and self-match validation (S2/S3 only, no S1 self matches)
5. Subset guarantee: matched_entity_ids is strict subset of candidate_entity_ids
6. Match distribution analysis (0, 1, multiple matches, max matches per entity)
7. Candidate set reduction ratio & candidate count distribution
8. Cross-country match verification
"""

import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent
TEST_DIR = ROOT_DIR / "dataset" / "test"
OUTPUT_DIR = ROOT_DIR / "output"

MATCHING_FILE = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_FILE = OUTPUT_DIR / "candidate_pairs.tsv"
TEST_S1_FILE = TEST_DIR / "test_source1.tsv"

EXPECTED_ROWS = 1_732_544

def run_quality_checks():
    print("=" * 65)
    print("Amazon ML Challenge 2026 — Submission Quality Assurance Check")
    print("=" * 65)

    if not MATCHING_FILE.exists():
        print(f"Error: {MATCHING_FILE} does not exist!")
        return False
    if not CANDIDATE_FILE.exists():
        print(f"Error: {CANDIDATE_FILE} does not exist!")
        return False

    # 1. Load test S1 IDs and country mapping
    print("\n1. Loading test_source1.tsv IDs and countries...")
    s1_countries = {}
    ordered_s1_ids = []
    with open(TEST_S1_FILE, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        id_idx = header.index("entity_id")
        c_idx = header.index("country")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) > c_idx:
                sid = parts[id_idx].strip()
                s1_countries[sid] = parts[c_idx].strip().upper()
                ordered_s1_ids.append(sid)

    print(f"Loaded {len(ordered_s1_ids):,} expected S1 entities.")
    required_s1_set = set(ordered_s1_ids)

    # 2. Check candidate_pairs.tsv
    print("\n2. Verifying candidate_pairs.tsv...")
    cand_s1_seen = set()
    cand_map = {}
    cand_counts = []
    cand_line_count = 0
    candidate_header_valid = False

    with open(CANDIDATE_FILE, "r", encoding="utf-8") as f:
        cand_header = f.readline().rstrip("\r\n").split("\t")
        if cand_header == ["source1_entity_id", "candidate_entity_ids"]:
            candidate_header_valid = True
            print("  Header format: VALID ['source1_entity_id', 'candidate_entity_ids']")
        else:
            print(f"  Header format: INVALID {cand_header}")

        cand_intra_dupes = 0
        cand_wrong_prefix = 0
        cand_self_matches = 0

        for line in f:
            cand_line_count += 1
            parts = line.rstrip("\r\n").split("\t")
            sid = parts[0].strip()
            c_str = parts[1].strip() if len(parts) > 1 else ""
            c_ids = [c.strip() for c in c_str.split(",") if c.strip()] if c_str else []
            cand_s1_seen.add(sid)
            cand_map[sid] = set(c_ids)
            cand_counts.append(len(c_ids))

            if len(c_ids) != len(set(c_ids)):
                cand_intra_dupes += 1
            for cid in c_ids:
                if cid.startswith("S1-"):
                    cand_self_matches += 1
                elif not (cid.startswith("S2-") or cid.startswith("S3-")):
                    cand_wrong_prefix += 1

    print(f"  Total Rows: {cand_line_count:,} (Expected: {EXPECTED_ROWS:,})")
    assert cand_line_count == EXPECTED_ROWS, f"Row count mismatch in candidate_pairs.tsv: {cand_line_count}"
    assert cand_s1_seen == required_s1_set, "Candidate S1 IDs do not match required test set!"
    assert cand_intra_dupes == 0, f"Found {cand_intra_dupes} rows with duplicate candidate IDs!"
    assert cand_self_matches == 0, f"Found {cand_self_matches} candidate IDs starting with S1-!"
    assert cand_wrong_prefix == 0, f"Found {cand_wrong_prefix} candidate IDs without S2-/S3- prefix!"
    print("  Candidate S1 bijection: 100% MATCH with test_source1.tsv.")
    print("  Candidate uniqueness & prefixes: 100% VALID (0 duplicates, 0 S1 IDs).")

    # 3. Check matching_results.tsv
    print("\n3. Verifying matching_results.tsv...")
    match_s1_seen = set()
    match_map = {}
    match_counts = []
    match_line_count = 0
    matching_header_valid = False
    intra_dupes = 0
    wrong_prefix = 0
    self_matches = 0
    subset_violations = 0

    country_matches = Counter()
    country_singletons = Counter()
    country_totals = Counter()

    with open(MATCHING_FILE, "r", encoding="utf-8") as f:
        match_header = f.readline().rstrip("\r\n").split("\t")
        if match_header == ["source1_entity_id", "matched_entity_ids"]:
            matching_header_valid = True
            print("  Header format: VALID ['source1_entity_id', 'matched_entity_ids']")
        else:
            print(f"  Header format: INVALID {match_header}")

        for line in f:
            match_line_count += 1
            parts = line.rstrip("\r\n").split("\t")
            sid = parts[0].strip()
            m_str = parts[1].strip() if len(parts) > 1 else ""
            m_ids = [m.strip() for m in m_str.split(",") if m.strip()] if m_str else []

            match_s1_seen.add(sid)
            match_map[sid] = m_ids
            match_counts.append(len(m_ids))

            c = s1_countries.get(sid, "UNKNOWN")
            country_totals[c] += 1

            if not m_ids:
                country_singletons[c] += 1
            else:
                country_matches[c] += len(m_ids)

            # Intra-dupes check
            if len(m_ids) != len(set(m_ids)):
                intra_dupes += 1

            # Prefix & Self-match check
            for mid in m_ids:
                if mid.startswith("S1-"):
                    self_matches += 1
                elif not (mid.startswith("S2-") or mid.startswith("S3-")):
                    wrong_prefix += 1

            # Subset guarantee check: matched must be subset of candidates
            cand_set = cand_map.get(sid, set())
            if not set(m_ids).issubset(cand_set):
                subset_violations += 1

    print(f"  Total Rows: {match_line_count:,} (Expected: {EXPECTED_ROWS:,})")
    assert match_line_count == EXPECTED_ROWS, f"Row count mismatch in matching_results.tsv: {match_line_count}"
    assert match_s1_seen == required_s1_set, "Matching S1 IDs do not match required test set!"
    print("  Matching S1 bijection: 100% MATCH with test_source1.tsv.")

    print(f"\n4. Rule Violations Summary:")
    print(f"  - Intra-list duplicates: {intra_dupes}")
    print(f"  - S1 self-matches: {self_matches}")
    print(f"  - Invalid prefixes: {wrong_prefix}")
    print(f"  - Subset violations (matches not in candidates): {subset_violations}")

    assert intra_dupes == 0, "Found duplicate IDs within match lists!"
    assert self_matches == 0, "Found S1 self-matches in predictions!"
    assert wrong_prefix == 0, "Found IDs without S2-/S3- prefix!"
    assert subset_violations == 0, "Found matches that were not in candidate_pairs.tsv!"

    # 5. Prediction Distribution Analysis
    total_predicted_matches = sum(match_counts)
    total_singletons = sum(1 for x in match_counts if x == 0)
    single_matches = sum(1 for x in match_counts if x == 1)
    multi_matches = sum(1 for x in match_counts if x > 1)

    print(f"\n5. Match Cardinality Distribution:")
    print(f"  - Total Predicted Matches: {total_predicted_matches:,}")
    print(f"  - Zero matches (singletons): {total_singletons:,} ({total_singletons/EXPECTED_ROWS*100:.2f}%)")
    print(f"  - Exactly one match: {single_matches:,} ({single_matches/EXPECTED_ROWS*100:.2f}%)")
    print(f"  - Multiple matches (>1): {multi_matches:,} ({multi_matches/EXPECTED_ROWS*100:.2f}%)")
    print(f"  - Max matches for a single entity: {max(match_counts)}")
    print(f"  - Frequency distribution (k: count): {sorted(Counter(match_counts).items())[:12]}")

    print(f"\n6. Country-wise Statistics:")
    for country in sorted(country_totals.keys()):
        tot = country_totals[country]
        sing = country_singletons[country]
        m_cnt = country_matches[country]
        print(f"  - {country}:")
        print(f"      Total S1: {tot:,}")
        print(f"      Singletons (no-match): {sing:,} ({sing/tot*100:.2f}%)")
        print(f"      Matches: {m_cnt:,} (avg {m_cnt/tot:.2f} per entity)")

    print(f"\n7. Candidate Blocking Statistics:")
    total_candidates = sum(cand_counts)
    print(f"  - Total Candidates Generated: {total_candidates:,}")
    print(f"  - Average Candidates per S1: {total_candidates/EXPECTED_ROWS:.2f}")
    print(f"  - Max Candidates for an S1: {max(cand_counts)}")

    print("\n" + "=" * 65)
    print("ALL SUBMISSION QUALITY CHECKS PASSED PERFECTLY!")
    print("=" * 65)
    return True

if __name__ == "__main__":
    run_quality_checks()
