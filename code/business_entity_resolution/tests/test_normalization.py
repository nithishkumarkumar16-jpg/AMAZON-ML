"""Unit tests for Phase 1 normalization, data loading, and evaluation.

Uses real examples discovered during dataset exploratory data analysis.
"""

import sys
from pathlib import Path
import pytest

# Add package source to path
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
from evaluation import compute_f05_score
from config import TRAIN_SOURCE1
from data_loader import stream_source_tsv, load_source_tsv


class TestNameNormalization:
    """Tests for business name cleaning and normalization."""

    def test_accent_and_unicode_removal(self):
        # Discovered in dataset inspection: 'Payne Énterprises', 'Lumay Bóral'
        assert remove_accents("Payne Énterprises") == "Payne Enterprises"
        assert remove_accents("Lumay Bóral") == "Lumay Boral"
        assert normalize_business_name("Payne Énterprises") == "payne"
        assert normalize_business_name("Lumay Bóral") == "lumay boral"

    def test_domain_and_url_normalization(self):
        # Discovered in dataset inspection: 'maurewilliamscolombier.com'
        assert normalize_domain_url("maurewilliamscolombier.com").strip() == "maurewilliamscolombier"
        assert normalize_business_name("maurewilliamscolombier.com") == "maurewilliamscolombier"
        assert normalize_business_name("https://www.maure-williams.org") == "maure williams"

    def test_legal_suffix_stripping_and_transposition(self):
        # Discovered in dataset inspection:
        # 'Maure Williams Colombier Inc'
        # '[Corp] Dick Regional Armada'
        # 'LLC Crystal Staffing Solutions'
        # 'CRYSTAL STAFFING SOLUTIONS-L.L.C.'
        assert normalize_business_name("Maure Williams Colombier Inc") == "maure williams colombier"
        assert normalize_business_name("[Corp] Dick Regional Armada") == "dick regional armada"
        assert normalize_business_name("LLC Crystal Staffing Solutions") == "crystal staffing solutions"
        assert normalize_business_name("CRYSTAL STAFFING SOLUTIONS-L.L.C.") == "crystal staffing solutions"
        assert normalize_business_name("Ss Food Private Limited") == "ss food"
        assert normalize_business_name("Raj Investments LLP") == "raj investments"
        assert normalize_business_name("Obsidian, [[LLC]]") == "obsidian"

    def test_dba_handling(self):
        # Discovered in dataset inspection: 'Korbrixx D.B.A. Obsidian, LLC'
        assert normalize_business_name("Korbrixx D.B.A. Obsidian, LLC") == "obsidian"
        assert normalize_business_name("Old Brand DBA New Venture") == "new venture"

    def test_conjunction_and_punctuation(self):
        # Discovered in dataset inspection: 'Chordia & Partners', 'Chordia + Pagnters - 7306204978'
        assert normalize_business_name("Chordia & Partners") == "chordia and"
        assert normalize_business_name("Chordia + Pagnters") == "chordia and pagnters"

    def test_empty_and_null_names(self):
        assert normalize_business_name("") == ""
        assert normalize_business_name(None) == ""
        assert normalize_business_name("   ") == ""


class TestAddressNormalization:
    """Tests for business address cleaning and abbreviation expansion."""

    def test_abbreviation_expansion(self):
        # Discovered in dataset inspection: '85 Wanye Ave, Ticonderoga Townshiip, NY'
        raw = "85 Wayne Ave, Ticonderoga, NY"
        norm = normalize_business_address(raw)
        assert "avenue" in norm
        assert "85" in norm
        assert "ticonderoga" in norm
        assert "ny" in norm

    def test_street_and_road_abbreviations(self):
        # Discovered in dataset inspection: '3315 FREMONT ST, PEORIA, IL'
        raw = "3315 FREMONT ST, PEORIA, IL"
        norm = normalize_business_address(raw)
        assert norm == "3315 fremont street peoria il"

    def test_null_token_removal(self):
        # Discovered in dataset inspection: '45ND TERRACE, null, KANSAS CITY, MO'
        # and '33466 WARWICK HILLS ROAD, <NULL>, YUCAIPA, CA'
        assert "null" not in normalize_business_address("45ND TERRACE, null, KANSAS CITY, MO")
        assert "<null>" not in normalize_business_address("33466 WARWICK HILLS ROAD, <NULL>, YUCAIPA, CA")

    def test_preserve_numeric_and_postal_tokens(self):
        # Discovered in dataset inspection:
        # Indian PIN code '9487203', door number 'Af-684', sector '1038 Sector 9'
        addr = "Af-684, Nandgram Near Mother India Public School. Ph. 989, 9487203, Ghaziabad"
        norm = normalize_business_address(addr)
        assert "684" in norm
        assert "9487203" in norm
        assert "989" in norm
        assert "ghaziabad" in norm

    def test_empty_and_missing_addresses(self):
        # Verified in dataset inspection: 610,000+ empty/null addresses
        assert normalize_business_address("") == ""
        assert normalize_business_address(None) == ""
        assert normalize_business_address("null") == ""
        assert normalize_business_address("<null>") == ""
        assert normalize_business_address("none") == ""
        assert normalize_business_address("   ") == ""


class TestOpenSetCountry:
    """Tests confirming country handling is open-set."""

    def test_open_set_handling(self):
        assert normalize_country("US") == "US"
        assert normalize_country("India") == "INDIA"
        # France is in the test set!
        assert normalize_country("France") == "FRANCE"
        # Any other unseen country label works transparently
        assert normalize_country("Germany") == "GERMANY"
        assert normalize_country("  Japan  ") == "JAPAN"
        assert normalize_country("") == ""
        assert normalize_country(None) == ""


class TestDataLoader:
    """Tests for TSV reading and chunk streaming."""

    def test_load_source_sample(self):
        df = load_source_tsv(TRAIN_SOURCE1, nrows=10)
        assert len(df) == 10
        assert list(df.columns) == ["entity_id", "business_name", "business_address", "country"]
        assert df["entity_id"].iloc[0].startswith("S1-")

    def test_stream_source_chunks(self):
        generator = stream_source_tsv(TRAIN_SOURCE1, chunksize=100, country_filter="US")
        first_chunk = next(generator)
        assert len(first_chunk) <= 100
        assert (first_chunk["country"] == "US").all()


class TestEvaluationMetric:
    """Tests confirming exact F_0.5 formula matching competition specification."""

    def test_readme_example(self):
        # Example from README.md line 204:
        # S1 matches [S2-00047, S3-00812]
        # Predicted [S2-00047, S2-00193, S3-00812]
        # Precision = 2/3, Recall = 1.0 -> F_0.5 = 0.7142857
        gt = {"S1-00001": {"S2-00047", "S3-00812"}}
        pred = {"S1-00001": {"S2-00047", "S2-00193", "S3-00812"}}
        score = compute_f05_score(gt, pred)
        assert abs(score - 0.7142857) < 1e-4

    def test_singleton_scoring(self):
        # Correctly predicting singleton gives 1.0
        gt = {"S1-00001": set()}
        pred = {"S1-00001": set()}
        assert compute_f05_score(gt, pred) == 1.0

        # Predicting a match for a singleton gives 0.0
        pred_wrong = {"S1-00001": {"S2-99999"}}
        assert compute_f05_score(gt, pred_wrong) == 0.0
