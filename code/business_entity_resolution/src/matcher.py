"""Deterministic precision-focused baseline matcher module.

Applies a calibrated similarity threshold to candidate pairs, supporting zero,
single, or multiple matches per S1 entity without artificial cardinality caps.
"""

from typing import Dict, List, Optional, Set, Tuple

try:
    from .features import compute_pair_features, score_candidate_pair
except ImportError:
    from features import compute_pair_features, score_candidate_pair


class BaselineMatcher:
    """Deterministic matcher for entity resolution baseline."""

    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold

    def match_candidates(
        self,
        s1_name: str,
        s1_addr: str,
        candidates: List[Tuple[str, str, str]],  # List of (cand_id, cand_name, cand_addr)
    ) -> Tuple[List[str], List[str]]:
        """Score candidate records against an S1 entity and apply threshold.

        Parameters
        ----------
        s1_name : str
            Normalized S1 business name.
        s1_addr : str
            Normalized S1 business address.
        candidates : list of (cand_id, cand_name, cand_addr)
            Candidate records retrieved from blocking.

        Returns
        -------
        candidate_ids : list of str
            Deduplicated candidate IDs evaluated for this entity.
        matched_ids : list of str
            Deduplicated matched IDs exceeding the decision threshold.
        """
        candidate_ids = []
        matched_ids = []

        seen_cand = set()
        for cand_id, cand_name, cand_addr in candidates:
            if cand_id in seen_cand:
                continue
            seen_cand.add(cand_id)
            candidate_ids.append(cand_id)

            feats = compute_pair_features(s1_name, s1_addr, cand_name, cand_addr)
            score = score_candidate_pair(feats)

            if score >= self.threshold:
                matched_ids.append(cand_id)

        return candidate_ids, matched_ids
