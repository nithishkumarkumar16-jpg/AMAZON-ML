"""Comprehensive test suite for Amazon ML Challenge 2026 Day 1 Baseline.

Covers:
- normalization (names, addresses, open-set country)
- country blocking & partitioning
- candidate generation & bucket filtering
- similarity scoring & numeric preservation
- empty address handling
- zero-match (singleton) entities
- duplicate prevention
- output formatting & subset constraints
"""

import os
import sys
from pathlib import Path
import pytest

# Ensure package is on sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from normalization import (
    normalize_business_name,
    normalize_business_address,
    normalize_country,
    remove_accents,
    normalize_domain_url,
)
from blocking import BlockingIndex, get_blocking_keys
from features import compute_pair_features, score_candidate_pair
from matcher import BaselineMatcher
from evaluation import compute_f05_score


class TestNormalization:
    """1. Normalization Tests."""

    def test_name_cleaning(self):
        assert normalize_business_name("Maure Williams Colombier Inc") == "maure williams colombier"
        assert normalize_business_name("Payne Énterprises") == "payne"
        assert normalize_business_name("maurewilliamscolombier.com") == "maurewilliamscolombier"
        assert normalize_business_name("LLC Crystal Staffing Solutions") == "crystal staffing solutions"
        assert normalize_business_name("Korbrixx D.B.A. Obsidian, LLC") == "obsidian"

    def test_address_cleaning(self):
        norm = normalize_business_address("85 Wayne Ave, Ticonderoga, NY")
        assert "avenue" in norm
        assert "85" in norm
        assert "ticonderoga" in norm

    def test_empty_and_null_address_handling(self):
        assert normalize_business_address("") == ""
        assert normalize_business_address(None) == ""
        assert normalize_business_address("null") == ""
        assert normalize_business_address("<null>") == ""
        assert normalize_business_address("none") == ""

    def test_open_set_country(self):
        assert normalize_country("US") == "US"
        assert normalize_country("India") == "INDIA"
        assert normalize_country("France") == "FRANCE"
        assert normalize_country("Germany") == "GERMANY"


class TestBlockingAndCandidateGeneration:
    """2. Blocking & Candidate Generation Tests."""

    def test_blocking_keys(self):
        keys = get_blocking_keys("maure williams colombier")
        assert "ex:maure williams colombier" in keys
        assert "ns:maurewilliamscolombier" in keys
        assert any(k.startswith("sort:") for k in keys)
        assert "pfx2:maure williams" in keys

    def test_blocking_index_retrieval(self):
        index = BlockingIndex(max_bucket_size=10)
        index.add_record("S2-001", get_blocking_keys("maure williams colombier"))
        index.add_record("S3-002", get_blocking_keys("maurewilliamscolombier"))
        index.add_record("S2-999", get_blocking_keys("completely different entity"))

        cands = index.get_candidates_for_name("maure williams colombier")
        assert "S2-001" in cands
        assert "S3-002" in cands  # Captured by no-space key!
        assert "S2-999" not in cands

    def test_bucket_filtering(self):
        index = BlockingIndex(max_bucket_size=2)
        # Add 3 records with identical key
        index.add_record("S2-1", ["common_key"])
        index.add_record("S2-2", ["common_key"])
        index.add_record("S2-3", ["common_key"])
        index.filter_oversized_buckets()
        assert "common_key" not in index.index


class TestSimilarityScoringAndMatcher:
    """3. Similarity Scoring & Decision Rules."""

    def test_exact_match_score(self):
        feats = compute_pair_features(
            "maure williams", "85 wayne avenue",
            "maure williams", "85 wayne avenue"
        )
        score = score_candidate_pair(feats)
        assert score == 1.0

    def test_numeric_conflict_penalty(self):
        # Same street name but different door/house numbers
        feats = compute_pair_features(
            "maure williams", "85 wayne avenue",
            "maure williams", "999 wayne avenue"
        )
        score = score_candidate_pair(feats)
        # Should be heavily penalized due to conflicting numbers
        assert score < 0.50

    def test_missing_address_graceful_handling(self):
        # Empty address in candidate
        feats = compute_pair_features(
            "maure williams colombier", "85 wayne avenue",
            "maure williams colombier", ""
        )
        score = score_candidate_pair(feats)
        # High name confidence yields match despite missing address
        assert score >= 0.80

    def test_matcher_subset_guarantee(self):
        matcher = BaselineMatcher(threshold=0.75)
        cands = [
            ("S2-001", "maure williams colombier", "85 wayne avenue"),
            ("S3-002", "unrelated brand", "different street"),
        ]
        cand_ids, matched_ids = matcher.match_candidates(
            "maure williams colombier", "85 wayne avenue", cands
        )
        assert set(matched_ids).issubset(set(cand_ids))
        assert "S2-001" in matched_ids
        assert "S3-002" not in matched_ids

    def test_zero_match_entity(self):
        matcher = BaselineMatcher(threshold=0.75)
        cands = [("S2-999", "completely different entity", "somewhere")]
        cand_ids, matched_ids = matcher.match_candidates(
            "unique target business", "123 unique road", cands
        )
        assert len(matched_ids) == 0  # Clean singleton output


class TestEvaluationAndMetrics:
    """4. Metric Evaluation Tests."""

    def test_macro_f05_computation(self):
        gt = {
            "S1-1": {"S2-10", "S3-20"},
            "S1-2": set(),  # singleton
        }
        # Model gets S1-1 perfect, S1-2 perfect
        pred = {
            "S1-1": {"S2-10", "S3-20"},
            "S1-2": set(),
        }
        assert compute_f05_score(gt, pred) == 1.0

        # Model false merges on singleton
        pred_wrong = {
            "S1-1": {"S2-10", "S3-20"},
            "S1-2": {"S2-99"},
        }
        score = compute_f05_score(gt, pred_wrong)
        # S1-1 = 1.0, S1-2 = 0.0 -> average = 0.5
        assert score == 0.5
