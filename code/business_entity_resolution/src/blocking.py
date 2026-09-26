"""Blocking and candidate generation module for Amazon ML Challenge 2026.

Implements multi-key blocking strategies on normalized business names within strict
country partitions. Produces a high-recall candidate set while avoiding Cartesian explosion.
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple


def get_blocking_keys(norm_name: str) -> List[str]:
    """Generate multi-angle blocking keys from a normalized business name.

    Keys generated:
    1. Exact normalized name (if len >= 2)
    2. Name without whitespace (bridges domain names like 'maurewilliamscolombier' and 'maure williams colombier')
    3. Sorted significant tokens (bridges word order permutations)
    4. First two tokens prefix (captures brand prefixes with modified legal tails)

    Parameters
    ----------
    norm_name : str
        Normalized business name string.

    Returns
    -------
    list of str
        Distinct non-empty blocking keys.
    """
    if not norm_name or len(norm_name) < 2:
        return []

    keys = []
    
    # 1. Exact normalized name
    keys.append(f"ex:{norm_name}")

    # 2. No-space representation (bridges domain names like maurewilliamscolombier.com)
    ns = norm_name.replace(" ", "")
    if len(ns) >= 3:
        keys.append(f"ns:{ns}")

    tokens = norm_name.split()

    # 3. Sorted significant tokens (if multiple words)
    if len(tokens) > 1:
        # Filter out 1-char tokens
        sig_tokens = [t for t in tokens if len(t) > 1]
        if len(sig_tokens) > 1:
            sorted_key = " ".join(sorted(sig_tokens))
            keys.append(f"sort:{sorted_key}")

    # 4. Lead prefix key (first 2 words if words >= 2, or first 6 chars if single long word)
    if len(tokens) >= 2:
        lead2 = " ".join(tokens[:2])
        if len(lead2) >= 4:
            keys.append(f"pfx2:{lead2}")
    elif len(tokens) == 1 and len(tokens[0]) >= 6:
        keys.append(f"pfx6:{tokens[0][:6]}")

    return list(dict.fromkeys(keys))  # Deduplicate while preserving order


class BlockingIndex:
    """Inverted index mapping blocking keys to secondary entity IDs (Source 2 or 3).
    
    Includes bucket size filtering to prevent noise explosion on high-frequency generic terms.
    """

    def __init__(self, max_bucket_size: int = 150):
        self.max_bucket_size = max_bucket_size
        self.index: Dict[str, List[str]] = defaultdict(list)

    def add_record(self, entity_id: str, keys: Iterable[str]):
        """Index an entity record by its precomputed blocking keys."""
        for key in keys:
            self.index[key].append(entity_id)

    def get_candidates_for_name(self, norm_name: str) -> Set[str]:
        """Retrieve candidate IDs for an S1 entity's normalized name."""
        candidates = set()
        for key in get_blocking_keys(norm_name):
            ids = self.index.get(key)
            if ids and len(ids) <= self.max_bucket_size:
                candidates.update(ids)
        return candidates

    def filter_oversized_buckets(self):
        """Purge keys that exceed the maximum bucket size to save memory and reduce noise."""
        oversized = [k for k, ids in self.index.items() if len(ids) > self.max_bucket_size]
        for k in oversized:
            del self.index[k]
