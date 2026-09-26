"""Feature calculation module for candidate pairs.

Computes lightweight, vectorized, and fast string similarity features across
normalized business names and addresses.
"""

from typing import Dict, Set, Tuple


def compute_pair_features(
    s1_name: str,
    s1_addr: str,
    cand_name: str,
    cand_addr: str,
) -> Dict[str, float]:
    """Compute pairwise similarity metrics between an S1 entity and a candidate.

    Parameters
    ----------
    s1_name : str
        Normalized S1 business name.
    s1_addr : str
        Normalized S1 business address.
    cand_name : str
        Normalized candidate business name.
    cand_addr : str
        Normalized candidate business address.

    Returns
    -------
    dict of str -> float
        Dictionary containing name and address similarity features.
    """
    feats = {}

    # --- 1. Name Similarities ---
    name_exact = 1.0 if s1_name and s1_name == cand_name else 0.0
    feats["name_exact"] = name_exact

    # Name without spaces (bridges domains and concatenated compound names)
    ns1 = s1_name.replace(" ", "")
    ns2 = cand_name.replace(" ", "")
    name_ns_match = 1.0 if len(ns1) >= 3 and ns1 == ns2 else 0.0
    feats["name_ns_match"] = name_ns_match

    # Name token sets
    tokens1 = set(s1_name.split())
    tokens2 = set(cand_name.split())
    n_union = len(tokens1 | tokens2)
    n_inter = len(tokens1 & tokens2)

    name_jaccard = (n_inter / n_union) if n_union > 0 else 0.0
    feats["name_jaccard"] = name_jaccard

    min_tokens = min(len(tokens1), len(tokens2))
    name_containment = (n_inter / min_tokens) if min_tokens > 0 else 0.0
    feats["name_containment"] = name_containment

    # Name length difference ratio
    max_len = max(len(s1_name), len(cand_name), 1)
    feats["name_len_diff"] = abs(len(s1_name) - len(cand_name)) / max_len

    # --- 2. Address Similarities ---
    addr_missing = 1.0 if (not s1_addr or not cand_addr) else 0.0
    feats["addr_missing"] = addr_missing

    if addr_missing == 0.0:
        feats["addr_exact"] = 1.0 if s1_addr == cand_addr else 0.0

        addr_tok1 = set(s1_addr.split())
        addr_tok2 = set(cand_addr.split())
        a_union = len(addr_tok1 | addr_tok2)
        a_inter = len(addr_tok1 & addr_tok2)
        feats["addr_jaccard"] = (a_inter / a_union) if a_union > 0 else 0.0

        # Numeric / Postal tokens (PIN codes, house numbers, sector numbers)
        nums1 = {t for t in addr_tok1 if any(c.isdigit() for c in t)}
        nums2 = {t for t in addr_tok2 if any(c.isdigit() for c in t)}
        if nums1 and nums2:
            num_inter = len(nums1 & nums2)
            num_union = len(nums1 | nums2)
            feats["numeric_jaccard"] = num_inter / num_union
            feats["numeric_conflict"] = 1.0 if num_inter == 0 else 0.0
        elif not nums1 and not nums2:
            feats["numeric_jaccard"] = 1.0
            feats["numeric_conflict"] = 0.0
        else:
            # One has numbers, the other doesn't
            feats["numeric_jaccard"] = 0.0
            feats["numeric_conflict"] = 0.0
    else:
        feats["addr_exact"] = 0.0
        feats["addr_jaccard"] = 0.0
        feats["numeric_jaccard"] = 0.0
        feats["numeric_conflict"] = 0.0

    return feats


def score_candidate_pair(feats: Dict[str, float]) -> float:
    """Calculate a precision-calibrated composite match score (0.0 to 1.0).

    Weights precision heavily to align with the F_0.5 evaluation metric:
    - If addresses exist and confirm each other, rewards the pair.
    - If addresses heavily conflict on numbers (different street/PIN code), heavily penalizes.
    - If addresses are missing, requires high name confidence.
    """
    name_score = max(
        feats["name_exact"],
        feats["name_ns_match"],
        feats["name_jaccard"] * 0.95,
        feats["name_containment"] * 0.85,
    )

    # Penalize large length differences when not an exact containment
    if feats["name_exact"] == 0.0 and feats["name_ns_match"] == 0.0:
        name_score -= feats["name_len_diff"] * 0.15

    name_score = max(0.0, min(1.0, name_score))

    if feats["addr_missing"] == 0.0:
        # Both addresses exist
        if feats.get("numeric_conflict", 0.0) == 1.0:
            # Fatal address conflict: same street but different house/door/PIN number
            addr_score = feats["addr_jaccard"] * 0.10
            composite = (0.50 * name_score + 0.50 * addr_score) * 0.65
        else:
            addr_score = max(
                feats["addr_exact"],
                0.6 * feats["addr_jaccard"] + 0.4 * feats["numeric_jaccard"],
            )
            composite = 0.65 * name_score + 0.35 * addr_score
    else:
        # Address is missing in S2 or S3
        # In our EDA, true matches with missing addresses had identical or nearly identical names
        if name_score >= 0.85:
            composite = name_score * 0.90  # Minor uncertainty discount
        else:
            composite = name_score * 0.60  # Require high name match when address absent

    return max(0.0, min(1.0, composite))
