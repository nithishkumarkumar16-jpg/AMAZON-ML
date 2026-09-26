# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

A production-grade, memory-efficient Entity Resolution (ER) system designed to link business records across three noisy independent data sources (Source 1 reference, Source 2, and Source 3) under strict non-external-lookup constraints.

---

## 1. Directory Structure

```
code/
└── business_entity_resolution/
    ├── src/
    │   ├── __init__.py
    │   ├── config.py             # File paths, constants, open-set configurations
    │   ├── data_loader.py        # Memory-efficient & chunked TSV reading with pandas
    │   ├── normalization.py      # Unicode, legal suffix, URL, and address normalizers
    │   ├── blocking.py           # Multi-key candidate generation (Phase 2)
    │   ├── features.py           # Pairwise string similarity feature engineering (Phase 2/3)
    │   ├── matcher.py            # Precision-calibrated classifier (Phase 3)
    │   ├── evaluation.py         # Exact macro F_0.5 evaluation implementation
    │   └── predict.py            # End-to-end inference and submission export
    ├── README.md                 # Reproduction instructions and architecture
    └── requirements.txt          # Pinned environment dependencies
```

---

## 2. Setup & Environment

Python 3.8+ is supported. Install dependencies:

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

---

## 3. Phase 1 Implementation Highlights

1. **TSV Streaming & Memory Efficiency:**
   - Multi-million-row TSVs (~26.4M rows total) are ingested via chunked streaming (`stream_source_tsv`) and strict `string` dtypes to prevent memory crashes.
   - `keep_default_na=False` prevents empty strings from becoming float `NaN` or string `"nan"`.
2. **Business Name Normalization:**
   - Unicode NFKD decomposition strips accents and diacritics (`Énterprises` → `enterprises`, `Bóral` → `boral`).
   - Domain URL normalization strips protocols (`https://`), prefixes (`www.`), and TLDs (`.com`, `.in`, `.org`).
   - DBA (`Doing Business As`) resolution focuses on the operative trade name.
   - Robust legal suffix standardization (`pvt ltd`, `llc`, `inc`, `corp`, `limited`, `enterprises`) whether placed as prefix (`[Corp] Dick Regional`), suffix, or infix.
   - Conjunction standardizations (`&`, `+` → `and`) and punctuation collapsing.
3. **Business Address Normalization:**
   - Municipal abbreviation expansions (`st` → `street`, `rd` → `road`, `ave` → `avenue`, `fl` → `floor`).
   - Null-token scrubbing (`null`, `<null>`, `none` → empty).
   - Strict preservation of critical numeric and postal tokens (PIN codes, ZIP codes, street numbers).
4. **Open-Set Country Handling:**
   - `normalize_country` treats country as an open-set string. It dynamically handles `US`, `India`, `France`, and any future country label without hard-coding assumptions.
5. **Exact Metric Alignment:**
   - `evaluation.py` implements the exact challenge Macro $F_{0.5}$ metric with singleton credit handling.

---

## 4. Running Tests & Pipeline

### Unit Tests
```bash
pytest code/business_entity_resolution/tests/test_pipeline.py -v
```

### Day 1 Baseline Inference
Generate the full submission files (`matching_results.tsv` and `candidate_pairs.tsv`):
```bash
python code/business_entity_resolution/src/predict.py
```

### Official Submission Validation
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

---

## 5. Day 1 Baseline Methodology & Results

### Multi-Key Blocking (Candidate Generation)
1. **Name Prefix Key**: First 6 characters of normalized name.
2. **First Word Key**: First significant token.
3. **Phonetic Double Metaphone Key**: Phonetic key for leading word tokens.
4. **Hard Country Blocking**: Guaranteed zero cross-country candidate leakage.

### Precision-Calibrated Matcher
- **Macro $F_{0.5}$ metric**: Weighs Precision twice as heavily as Recall ($\beta = 0.5$).
- **Features**: Exact name equality, token Jaccard similarity, character 3-gram dice similarity, Levenshtein edit ratio, address word overlap, address digit consistency.
- **Threshold Calibration**: Evaluated across decision thresholds $[0.60, 0.85]$ on a 5,000-entity validation split across US and India (17,361 ground-truth pairs).
  - Selected Threshold: **0.75** (Macro $F_{0.5} = 0.5933$, candidate recall ceiling = 59.04%).

### Test Set Execution Metrics (1,732,544 S1 Entities)
- **Total Pipeline Runtime**: 5,067.07s
- **Total Candidates Generated**: 52,592,603 (30.36 candidates/entity avg, max 189)
- **Total Predicted Matches**: 4,215,309 (2.43 matches/entity avg, max 38)
- **Zero Matches (Singletons)**: 251,667 (14.53% vs ~15.9% in training ground truth)
- **Single Matches**: 328,510 (18.96%)
- **Multiple Matches**: 1,152,367 (66.51%)
- **Country Distribution**:
  - **France**: 259,452 entities | 688,608 matches (2.65 avg) | 28,679 singletons (11.05%)
  - **India**: 809,986 entities | 2,014,365 matches (2.49 avg) | 98,990 singletons (12.22%)
  - **US**: 663,106 entities | 1,512,336 matches (2.28 avg) | 123,998 singletons (18.70%)

### Official Validator Verification
```
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  valid S2/S3 match IDs: 9969589
  matching_results.tsv: 1732544 rows (251667 empty, 1480877 non-empty).
  candidate_pairs.tsv: 1732544 rows (84794 empty, 1647750 non-empty).
PASS — no blocking issues found. Safe to submit.
```
- **0** missing entities (100% bijection with `test_source1.tsv`).
- **0** intra-row duplicates.
- **0** S1 self-matches.
- **0** subset violations (`matched_entity_ids` $\subseteq$ `candidate_entity_ids`).
- **0** invalid IDs across all 9,969,589 candidate targets checked.
