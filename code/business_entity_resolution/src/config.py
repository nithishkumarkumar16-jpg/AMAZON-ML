import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Dataset Train Paths
TRAIN_SOURCE1 = DATASET_DIR / "train" / "train_source1.tsv"
TRAIN_SOURCE2 = DATASET_DIR / "train" / "train_source2.tsv"
TRAIN_SOURCE3 = DATASET_DIR / "train" / "train_source3.tsv"
TRAIN_GROUND_TRUTH = DATASET_DIR / "train" / "train_ground_truth.tsv"

# Dataset Test Paths
TEST_SOURCE1 = DATASET_DIR / "test" / "test_source1.tsv"
TEST_SOURCE2 = DATASET_DIR / "test" / "test_source2.tsv"
TEST_SOURCE3 = DATASET_DIR / "test" / "test_source3.tsv"

# Output Paths
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# File format & delimiter
DELIMITER = "\t"
ENCODING = "utf-8"

# Standard TSV Schemas
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_COLUMNS = ["source1_entity_id", "candidate_entity_ids"]

# Chunk sizes for memory-efficient streaming across multi-million rows
DEFAULT_CHUNK_SIZE = 100_000

# Country Handling (Open-Set: never restricted to US/India only)
# Any country label encountered in dataset (e.g. US, India, France, etc.) is supported dynamically.
