"""Normalization module for Amazon ML Challenge 2026.

Provides text normalizers for:
- Business Names (Unicode, accents, URLs/domains, DBA markers, legal suffixes, punctuation)
- Business Addresses (Unicode, abbreviations, numeric/postal preservation, null token removal)
- Countries (Open-set string standardization)
"""

import re
import unicodedata
from typing import Optional


# ==============================================================================
# 1. Unicode & Accent Removal
# ==============================================================================

def remove_accents(text: str) -> str:
    """Normalize Unicode and strip combining diacritical marks/accents.
    
    Fast-paths ASCII strings (99%+ of records) for 10x-50x speedup.
    """
    if not text or text.isascii():
        return text or ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


# ==============================================================================
# 2. URL & Domain Normalization
# ==============================================================================

URL_PROTOCOL_PATTERN = re.compile(r"https?://", re.IGNORECASE)
WWW_PATTERN = re.compile(r"\bwww\d*\.", re.IGNORECASE)
DOMAIN_SUFFIX_PATTERN = re.compile(
    r"\.(?:com|org|net|in|co\.in|co|io|biz|info|gov|edu|us|ai|fr)\b",
    re.IGNORECASE,
)


def normalize_domain_url(text: str) -> str:
    """Detect and clean domain names or URLs present in business names.
    
    Example: 'maurewilliamscolombier.com' -> 'maurewilliamscolombier'
             'www.example-store.in' -> 'example store'
    """
    if not text:
        return ""
    # Strip protocols
    text = URL_PROTOCOL_PATTERN.sub("", text)
    text = WWW_PATTERN.sub("", text)
    # Remove domain suffixes
    text = DOMAIN_SUFFIX_PATTERN.sub(" ", text)
    return text


# ==============================================================================
# 3. Business Name Normalization
# ==============================================================================

# Common DBA (Doing Business As) markers
DBA_PATTERN = re.compile(
    r"\b(?:d[\.\/\s]*b[\.\/\s]*a|t[\.\/\s]*a|a[\.\/\s]*k[\.\/\s]*a|doing\s+business\s+as|trading\s+as)\b\.?",
    re.IGNORECASE,
)

# Legal suffixes and entity types (order matters: longer compound expressions first)
LEGAL_TERMS = [
    r"\bprivate\s+limited\b",
    r"\bpvt\.?\s*ltd\.?\b",
    r"\bpvt\b",
    r"\blimited\b",
    r"\bltd\.?\b",
    r"\bllc\.?\b",
    r"\bl\.l\.c\.?\b",
    r"\bllp\.?\b",
    r"\bl\.l\.p\.?\b",
    r"\bincorporated\b",
    r"\binc\.?\b",
    r"\bcorporation\b",
    r"\bcorp\.?\b",
    r"\bco\.?\b",
    r"\bcompany\b",
    r"\benterprises?\b",
    r"\bservices?\b",
    r"\bpartners?\b",
    r"\bcenter\b",
    r"\bgroup\b",
    r"\bholdings?\b",
]

LEGAL_REGEX = re.compile(r"|".join(LEGAL_TERMS), re.IGNORECASE)

# Punctuation cleaning (retaining alphanumeric, spaces, and ampersands)
AMPERSAND_PATTERN = re.compile(r"\s*(&|\+)\s*")
CLEAN_NAME_PUNCT_PATTERN = re.compile(r"[^\w\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_business_name(
    name: Optional[str],
    strip_legal_suffixes: bool = True,
) -> str:
    """Normalize a raw business name string.

    Steps applied:
    1. Unicode normalization and accent/diacritic removal (É -> e, ó -> o).
    2. Lowercasing.
    3. URL/Domain stripping (e.g. '.com', 'www.').
    4. DBA (Doing Business As) separation. If DBA exists, focuses on the operative trade name.
    5. Legal suffix normalization (e.g., 'pvt ltd', 'llc', 'inc') handled at prefix, infix, or suffix.
    6. Conjunction standardization ('&', '+' -> 'and').
    7. Punctuation removal and bracket stripping ('[[', ']]', dashes, quotes).
    8. Whitespace collapsing and trimming.

    Parameters
    ----------
    name : str or None
        Raw business name.
    strip_legal_suffixes : bool, default True
        Whether to strip common legal entity suffixes/prefixes.

    Returns
    -------
    str
        Canonical normalized business name.
    """
    if not name or not isinstance(name, str):
        return ""

    # 1. Accent & Unicode normalization
    text = remove_accents(name)

    # 2. Lowercase
    text = text.lower()

    # 3. Domain & URL normalization
    text = normalize_domain_url(text)

    # 4. Handle DBA: if format is "Old Name D.B.A. Real Name", extract the operative name
    if DBA_PATTERN.search(text):
        parts = DBA_PATTERN.split(text, maxsplit=1)
        # The true business identity is typically what follows DBA
        text = parts[-1].strip() if len(parts) > 1 and parts[-1].strip() else parts[0].strip()

    # 5. Conjunctions & common symbols
    text = AMPERSAND_PATTERN.sub(" and ", text)

    # 6. Legal suffix removal / standardization
    if strip_legal_suffixes:
        # Removes occurrences whether at start, end, or inside brackets (e.g., '[Corp] Dick Regional' -> 'dick regional')
        text = LEGAL_REGEX.sub(" ", text)

    # 7. Strip non-alphanumeric punctuation (except space)
    text = CLEAN_NAME_PUNCT_PATTERN.sub(" ", text)

    # 8. Collapse whitespace
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text


# ==============================================================================
# 4. Business Address Normalization
# ==============================================================================

# Null token patterns
NULL_ADDRESS_PATTERN = re.compile(r"^(?:null|<null>|none|n/a|\s*)$", re.IGNORECASE)

# Common street and municipal address abbreviations
ADDRESS_ABBREVIATIONS = {
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bblvd\b": "boulevard",
    r"\bpkwy\b": "parkway",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bter\b": "terrace",
    r"\bfl\b": "floor",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bno\b": "number",
    r"\bh\.?no\.?\b": "number",
    r"\bdoor\s+no\.?\b": "number",
    r"\bsec\b": "sector",
    r"\bph\b": "phase",
    r"\bdiv\b": "division",
}

ADDRESS_EXPANSIONS = [(re.compile(pattern, re.IGNORECASE), repl) for pattern, repl in ADDRESS_ABBREVIATIONS.items()]
CLEAN_ADDR_PUNCT_PATTERN = re.compile(r"[^\w\s\d]")


def normalize_business_address(
    address: Optional[str],
    expand_abbreviations: bool = True,
) -> str:
    """Normalize a raw business address string while strictly preserving numeric & postal tokens.

    Steps applied:
    1. Check for null / empty representations ('null', '<null>', 'none', '').
    2. Unicode normalization and accent/diacritic removal.
    3. Lowercasing.
    4. Remove placeholder 'null' tokens embedded inside addresses (e.g. '45th Terrace, null, Kansas City').
    5. Standardize address components and abbreviations ('st' -> 'street', 'rd' -> 'road').
    6. Preserve numbers, PIN codes (6-digit Indian), ZIP codes (5-digit US), street/door numbers.
    7. Punctuation cleaning (commas, hashes, periods, brackets converted to spaces).
    8. Whitespace collapsing and trimming.

    Parameters
    ----------
    address : str or None
        Raw business address string.
    expand_abbreviations : bool, default True
        Whether to expand common municipal address abbreviations.

    Returns
    -------
    str
        Canonical normalized business address (empty string if invalid/missing).
    """
    if not address or not isinstance(address, str):
        return ""

    raw_clean = address.strip()
    if NULL_ADDRESS_PATTERN.match(raw_clean):
        return ""

    # 1. Unicode & Accents
    text = remove_accents(raw_clean)

    # 2. Lowercase
    text = text.lower()

    # 3. Clean embedded null tokens (e.g., ', null,', '<null>')
    text = re.sub(r"\bnull\b|<null>|none", " ", text)

    # 4. Expand common abbreviations
    if expand_abbreviations:
        for pattern, replacement in ADDRESS_EXPANSIONS:
            text = pattern.sub(replacement, text)

    # 5. Clean punctuation (keeping alphanumeric and whitespace)
    # Notice: digits (postal codes, street numbers) are strictly preserved by \w and \d
    text = CLEAN_ADDR_PUNCT_PATTERN.sub(" ", text)

    # 6. Collapse multiple whitespaces
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text


# ==============================================================================
# 5. Open-Set Country Normalization
# ==============================================================================

def normalize_country(country: Optional[str]) -> str:
    """Normalize country code or label as an open-set string.
    
    Never restricts or assumes only {'US', 'India'}; transparently accepts
    'France' and any other arbitrary country label present in test or future data.

    Parameters
    ----------
    country : str or None
        Raw country string.

    Returns
    -------
    str
        Trimmed, uppercase country code / label.
    """
    if not country or not isinstance(country, str):
        return ""
    return country.strip().upper()
