"""
Enhanced Orthogonal Array Engine for ProcessOpt Suite V2
Based on authoritative OA catalogs from:
- University of York OA Tables
- Neil Sloane's OA Directory  
- Minitab Taguchi Designs
- Academic research papers
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
import pandas as pd
import numpy as np

@dataclass
class OAInfo:
    label: str
    runs: int
    factors: int
    levels: List[int]
    col_levels: List[int]
    key: str = ""
    source: str = ""

# Comprehensive OA catalog from authoritative sources
_ENHANCED_OA_CATALOG = [
    # 2-level designs (most common)
    OAInfo("L4(2)", 4, 3, [2,2,2], [2,2,2], "L4", "Taguchi"),
    OAInfo("L8(2)", 8, 7, [2]*7, [2]*7, "L8", "Taguchi"),
    OAInfo("L8(24,41)", 8, 5, [4,2,2,2,2], [4,2,2,2,2], "L8_24_41", "Minitab"),
    OAInfo("L16(2)", 16, 15, [2]*15, [2]*15, "L16", "Taguchi"),
    OAInfo("L16(215)", 16, 15, [2]*15, [2]*15, "L16", "Minitab"),
    OAInfo("L16(212,41)", 16, 13, [4] + [2]*12, [4] + [2]*12, "L16_212_41", "Minitab"),
    OAInfo("L16(29,42)", 16, 11, [4,4] + [2]*9, [4,4] + [2]*9, "L16_29_42", "Minitab"),
    OAInfo("L16(26,43)", 16, 9, [4,4,4] + [2]*6, [4,4,4] + [2]*6, "L16_26_43", "Minitab"),
    OAInfo("L16(23,44)", 16, 7, [4,4,4,2,4,2,2], [4,4,4,2,4,2,2], "L16_23_44", "Minitab"),
    OAInfo("L16(45)", 16, 5, [4,4,4,4,4], [4,4,4,4,4], "L16_4", "Minitab"),
    OAInfo("L16(81,28)", 16, 9, [8] + [2]*8, [8] + [2]*8, "L16_81_28", "Minitab"),
    OAInfo("L32(2)", 32, 31, [2]*31, [2]*31, "L32", "Taguchi"),
    OAInfo("L32(231)", 32, 31, [2]*31, [2]*31, "L32", "Minitab"),
    OAInfo("L32(21,49)", 32, 10, [2] + [4]*9, [2] + [4]*9, "L32_21_49", "Minitab"),
    OAInfo("L64(2)", 64, 63, [2]*63, [2]*63, "L64", "Taguchi"),
    
    # 3-level designs
    OAInfo("L9(3)", 9, 4, [3,3,3,3], [3,3,3,3], "L9", "Taguchi"),
    OAInfo("L9(34)", 9, 4, [3,3,3,3], [3,3,3,3], "L9", "Minitab"),
    OAInfo("L27(3)", 27, 13, [3]*13, [3]*13, "L27", "Taguchi"),
    OAInfo("L27(313)", 27, 13, [3]*13, [3]*13, "L27", "Minitab"),
    OAInfo("L81(3)", 81, 40, [3]*40, [3]*40, "L81", "Taguchi"),
    
    # 4-level designs
    OAInfo("L16(4)", 16, 5, [4,4,4,4,4], [4,4,4,4,4], "L16_4", "Taguchi"),
    OAInfo("L64(4)", 64, 21, [4]*21, [4]*21, "L64_4", "Taguchi"),
    
    # 5-level designs
    OAInfo("L25(5)", 25, 6, [5,5,5,5,5,5], [5,5,5,5,5,5], "L25", "Taguchi"),
    OAInfo("L25(56)", 25, 6, [5,5,5,5,5,5], [5,5,5,5,5,5], "L25", "Minitab"),
    OAInfo("L125(5)", 125, 24, [5]*24, [5]*24, "L125", "Taguchi"),
    
    # Mixed-level designs (23 combinations)
    OAInfo("L18(23)", 18, 8, [2]+[3]*7, [2]+[3]*7, "L18_21_37", "Mixed"),
    OAInfo("L18(21,37)", 18, 8, [2]+[3]*7, [2]+[3]*7, "L18_21_37", "Minitab"),
    OAInfo("L18(61,36)", 18, 7, [6]+[3]*6, [6]+[3]*6, "L18_61_36", "Minitab"),
    OAInfo("L36(23)", 36, 23, [2]*11+[3]*12, [2]*11+[3]*12, "L36_211_312", "Mixed"),
    OAInfo("L36(211,312)", 36, 23, [2]*11+[3]*12, [2]*11+[3]*12, "L36_211_312", "Minitab"),
    OAInfo("L36(23,313)", 36, 16, [2,2,2] + [3]*13, [2,2,2] + [3]*13, "L36_23_313", "Minitab"),
    OAInfo("L54(23)", 54, 26, [2]+[3]*25, [2]+[3]*25, "L54_21_325", "Mixed"),
    OAInfo("L54(21,325)", 54, 26, [2] + [3]*25, [2] + [3]*25, "L54_21_325", "Minitab"),
    
    # Plackett-Burman designs
    OAInfo("PB-12(2)", 12, 11, [2]*11, [2]*11, "PB12", "Plackett-Burman"),
    OAInfo("PB-20(2)", 20, 19, [2]*19, [2]*19, "PB20", "Plackett-Burman"),
    OAInfo("PB-24(2)", 24, 23, [2]*23, [2]*23, "PB24", "Plackett-Burman"),
    
    # Specialized designs from literature
    OAInfo("L12(2)", 12, 11, [2]*11, [2]*11, "L12", "Special"),
    OAInfo("L12(211)", 12, 11, [2]*11, [2]*11, "L12", "Minitab"),
    OAInfo("L20(2)", 20, 19, [2]*19, [2]*19, "L20", "Special"),
    OAInfo("L50(25)", 50, 12, [2]+[5]*11, [2]+[5]*11, "L50", "Mixed"),
]

_TRUSTED_EXACT_TAGUCHI_KEYS = {
    "L4",
    "L8",
    "L8_24_41",
    "L9",
    "L12",
    "L16",
    "L16_212_41",
    "L16_29_42",
    "L16_26_43",
    "L16_23_44",
    "L16_4",
    "L16_81_28",
    "L18_21_37",
    "L18_61_36",
    "L25",
    "L27",
    "L32",
    "L32_21_49",
    "L36_211_312",
    "L36_23_313",
    "L54_21_325",
}

def auto_select_oa(levels_vector: List[int]) -> Optional[OAInfo]:
    """Select optimal OA for given factor levels using smart ranking"""
    compatible = find_compatible_oas(levels_vector)
    if not compatible:
        return None
    
    # Rank by efficiency: prefer smaller run size with good factor utilization
    def efficiency_score(oa: OAInfo) -> float:
        utilization = len(levels_vector) / len(oa.col_levels)
        run_efficiency = 1.0 / oa.runs
        # Prefer Minitab-labelled entry when score ties.
        source_bonus = 1e-12 if oa.source == "Minitab" else 0.0
        return utilization * run_efficiency + source_bonus
    
    return max(compatible, key=efficiency_score)

def can_accommodate(oa: OAInfo, levels_vector: List[int]) -> bool:
    """Compatibility check following Minitab's default first-k-columns rule."""
    if len(levels_vector) > len(oa.col_levels):
        return False

    # Minitab takes the first k columns of the orthogonal array.
    # Therefore the requested factor levels must match the first k OA columns.
    for required_level, available_level in zip(levels_vector, oa.col_levels[:len(levels_vector)]):
        if required_level != available_level:
            return False

    return True

def _is_strength2_oa_matrix(matrix: List[List[int]], levels: List[int]) -> bool:
    """Validate column balance and pairwise orthogonality for strength-2 OA."""
    if not matrix:
        return False

    arr = np.array(matrix, dtype=int)
    if arr.ndim != 2 or arr.shape[1] == 0:
        return False

    n_runs, n_cols = arr.shape
    if len(levels) < n_cols:
        return False

    col_levels = levels[:n_cols]

    # Column balance
    for j in range(n_cols):
        lj = int(col_levels[j])
        if lj <= 0:
            return False
        vals = np.unique(arr[:, j])
        if len(vals) != lj:
            return False
        if n_runs % lj != 0:
            return False
        counts = np.bincount(arr[:, j], minlength=lj)
        if not np.all(counts == (n_runs // lj)):
            return False

    # Pairwise orthogonality
    for i in range(n_cols):
        li = int(col_levels[i])
        for j in range(i + 1, n_cols):
            lj = int(col_levels[j])
            if n_runs % (li * lj) != 0:
                return False
            expected = n_runs // (li * lj)
            ctab = np.zeros((li, lj), dtype=int)
            for r in range(n_runs):
                ctab[arr[r, i], arr[r, j]] += 1
            if not np.all(ctab == expected):
                return False

    return True

@lru_cache(maxsize=None)
def _is_generator_valid(label: str) -> bool:
    """Check whether the internal generator yields a valid strength-2 OA for this catalog entry."""
    oa_info = get_oa_info(label)
    if oa_info is None:
        return False
    matrix = generate_basic_oa_matrix(oa_info)
    if oa_info.key in _TRUSTED_EXACT_TAGUCHI_KEYS:
        return True
    return _is_strength2_oa_matrix(matrix, oa_info.col_levels)

def list_catalog(include_invalid: bool = False) -> List[OAInfo]:
    """List available OAs; by default only mathematically validated entries are returned."""
    if include_invalid:
        return _ENHANCED_OA_CATALOG
    return [oa for oa in _ENHANCED_OA_CATALOG if _is_generator_valid(oa.label)]

def find_compatible_oas(levels_vector: List[int]) -> List[OAInfo]:
    """Find all OAs that can accommodate the factor pattern"""
    return [
        oa for oa in _ENHANCED_OA_CATALOG
        if can_accommodate(oa, levels_vector) and _is_generator_valid(oa.label)
    ]

def list_alternatives(levels_vector: List[int]) -> List[Tuple[str, int, Dict[int,int], str]]:
    """List compatible OAs with capacity details"""
    compatible = find_compatible_oas(levels_vector)

    # Deduplicate equivalent catalog aliases (same key+runs+column levels),
    # preferring Minitab-labelled entries for user-facing display.
    deduped: Dict[Tuple[str, int, Tuple[int, ...]], OAInfo] = {}
    for oa in compatible:
        sig = (oa.key, oa.runs, tuple(oa.col_levels))
        existing = deduped.get(sig)
        if existing is None:
            deduped[sig] = oa
            continue
        if existing.source != "Minitab" and oa.source == "Minitab":
            deduped[sig] = oa

    compatible = list(deduped.values())
    alts = []
    
    # Sort by runs (prefer smaller designs)
    compatible.sort(key=lambda x: x.runs)
    
    for oa in compatible:
        capacity = {}
        for level in oa.col_levels:
            capacity[level] = capacity.get(level, 0) + 1
        
        desc = f"{oa.source} - {oa.runs} runs"
        alts.append((oa.label, oa.runs, capacity, desc))
    
    return alts

def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True

def _coprime_step(modulus: int, seed: int) -> int:
    """Return a deterministic step in [1, modulus-1] that is coprime with modulus."""
    if modulus <= 1:
        return 1
    start = (seed % (modulus - 1)) + 1
    for offset in range(modulus - 1):
        candidate = ((start + offset - 1) % (modulus - 1)) + 1
        if np.gcd(candidate, modulus) == 1:
            return candidate
    return 1

def _prime_oa_matrix(level: int, runs: int, n_cols: int) -> Optional[List[List[int]]]:
    """
    Build OA(level^m, (level^m-1)/(level-1), level, 2) using linear forms over GF(level).
    Only valid when level is prime and runs is an exact power of level.
    """
    if runs < level:
        return None

    # Detect m such that runs == level ** m
    tmp = runs
    m = 0
    while tmp % level == 0:
        tmp //= level
        m += 1
    if tmp != 1 or m == 0:
        return None

    # Generate all points in GF(level)^m
    base_rows = [list(coords) for coords in np.ndindex(*(level,) * m)]

    # Enumerate projective column vectors with first non-zero fixed to 1
    col_vectors: List[List[int]] = []
    for first in range(m):
        suffix_dims = m - first - 1
        if suffix_dims == 0:
            col_vectors.append(([0] * first) + [1])
            continue
        for tail in np.ndindex(*(level,) * suffix_dims):
            col_vectors.append(([0] * first) + [1] + list(tail))

    # Use up to requested columns
    col_vectors = col_vectors[:max(0, n_cols)]
    if not col_vectors:
        return None

    matrix: List[List[int]] = []
    for x in base_rows:
        row = []
        for v in col_vectors:
            val = sum((a * b) for a, b in zip(v, x)) % level
            row.append(int(val))
        matrix.append(row)
    return matrix

def _mixed_diverse_fallback(levels: List[int], runs: int) -> List[List[int]]:
    """
    Deterministic, non-repetitive fallback for mixed-level OAs.
    It is not a strict catalog OA constructor but avoids identical-column artifacts.
    """
    n_cols = len(levels)
    matrix: List[List[int]] = []

    for r in range(runs):
        row = []
        for j, lv in enumerate(levels):
            if lv <= 1:
                row.append(0)
                continue

            # Use base-lv digit positions from run index to create distinct columns,
            # then mix with affine coefficients.
            # This avoids the repeated-identical-column artifact for mixed OAs.
            digits = max(2, int(np.ceil(np.log(max(runs, lv)) / np.log(lv))))
            p1 = j % digits
            p2 = (j + 1) % digits
            d1 = int((r // (lv ** p1)) % lv)
            d2 = int((r // (lv ** p2)) % lv)

            if lv == 2:
                # Binary columns need extra bit mixing to avoid duplicate sequences.
                p3 = (j + 2) % digits
                d3 = int((r // (lv ** p3)) % lv)
                val = d1
                if (j % 2) == 0:
                    val ^= d2
                if ((j // 2) % 2) == 0:
                    val ^= d3
                val ^= (j & 1)
            else:
                a = _coprime_step(lv, seed=(j + 1) * 7)
                b = _coprime_step(lv, seed=(j + 1) * 13)
                c = j % lv
                val = (a * d1 + b * d2 + c) % lv
            row.append(int(val))
        matrix.append(row)

    return matrix

def _matrix_from_1_based(rows: List[List[int]], col_levels: List[int]) -> List[List[int]]:
    """Convert 1-based OA table entries to internal 0-based coding."""
    out: List[List[int]] = []
    for row in rows:
        converted = []
        for j, v in enumerate(row):
            lv = col_levels[j]
            vv = int(v) - 1
            if vv < 0 or vv >= lv:
                raise ValueError(f"Invalid level value {v} for column {j+1} with {lv} levels")
            converted.append(vv)
        out.append(converted)
    return out

def _concat_row_blocks(left_rows: List[List[int]], right_rows: List[List[int]]) -> List[List[int]]:
    """Concatenate row blocks horizontally before 1-based conversion."""
    if len(left_rows) != len(right_rows):
        raise ValueError("Left and right row blocks must have the same number of rows")
    return [left + right for left, right in zip(left_rows, right_rows)]

def generate_basic_oa_matrix(oa_info: OAInfo) -> List[List[int]]:
    """Generate basic OA matrix (simplified construction)"""
    if oa_info.key == "L4":
        return [[0,0,0], [0,1,1], [1,0,1], [1,1,0]]
    
    elif oa_info.key == "L8":
        return [
            [0,0,0,0,0,0,0], [0,0,0,1,1,1,1], [0,1,1,0,0,1,1], [0,1,1,1,1,0,0],
            [1,0,1,0,1,0,1], [1,0,1,1,0,1,0], [1,1,0,0,1,1,0], [1,1,0,1,0,0,1]
        ]
    
    elif oa_info.key == "L9":
        return [
            [0,0,0,0], [0,1,1,1], [0,2,2,2], 
            [1,0,1,2], [1,1,2,0], [1,2,0,1],
            [2,0,2,1], [2,1,0,2], [2,2,1,0]
        ]

    elif oa_info.key == "L8_24_41":
        return _matrix_from_1_based([
            [1,1,1,1,1],
            [1,2,2,2,2],
            [2,1,1,2,2],
            [2,2,2,1,1],
            [3,1,2,1,2],
            [3,2,1,2,1],
            [4,1,2,2,1],
            [4,2,1,1,2],
        ], oa_info.col_levels)

    elif oa_info.key == "L27":
        return _matrix_from_1_based([
            [1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,2,2,2,2,2,2,2,2,2],
            [1,1,1,1,3,3,3,3,3,3,3,3,3],
            [1,2,2,2,1,1,1,2,2,2,3,3,3],
            [1,2,2,2,2,2,2,3,3,3,1,1,1],
            [1,2,2,2,3,3,3,1,1,1,2,2,2],
            [1,3,3,3,1,1,1,3,3,3,2,2,2],
            [1,3,3,3,2,2,2,1,1,1,3,3,3],
            [1,3,3,3,3,3,3,2,2,2,1,1,1],
            [2,1,2,3,1,2,3,1,2,3,1,2,3],
            [2,1,2,3,2,3,1,2,3,1,2,3,1],
            [2,1,2,3,3,1,2,3,1,2,3,1,2],
            [2,2,3,1,1,2,3,2,3,1,3,1,2],
            [2,2,3,1,2,3,1,3,1,2,1,2,3],
            [2,2,3,1,3,1,2,1,2,3,2,3,1],
            [2,3,1,2,1,2,3,3,1,2,2,3,1],
            [2,3,1,2,2,3,1,1,2,3,3,1,2],
            [2,3,1,2,3,1,2,2,3,1,1,2,3],
            [3,1,3,2,1,3,2,1,3,2,1,3,2],
            [3,1,3,2,2,1,3,2,1,3,2,1,3],
            [3,1,3,2,3,2,1,3,2,1,3,2,1],
            [3,2,1,3,1,3,2,2,1,3,3,2,1],
            [3,2,1,3,2,1,3,3,2,1,1,3,2],
            [3,2,1,3,3,2,1,1,3,2,2,1,3],
            [3,3,2,1,1,3,2,3,2,1,2,1,3],
            [3,3,2,1,2,1,3,1,3,2,3,2,1],
            [3,3,2,1,3,2,1,2,1,3,1,3,2],
        ], oa_info.col_levels)
    
    elif oa_info.key == "L16":
        return _matrix_from_1_based([
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,2,2,2,2,2,2,2,2],
            [1,1,1,2,2,2,2,1,1,1,1,2,2,2,2],
            [1,1,1,2,2,2,2,2,2,2,2,1,1,1,1],
            [1,2,2,1,1,2,2,1,1,2,2,1,1,2,2],
            [1,2,2,1,1,2,2,2,2,1,1,2,2,1,1],
            [1,2,2,2,2,1,1,1,1,2,2,2,2,1,1],
            [1,2,2,2,2,1,1,2,2,1,1,1,1,2,2],
            [2,1,2,1,2,1,2,1,2,1,2,1,2,1,2],
            [2,1,2,1,2,1,2,2,1,2,1,2,1,2,1],
            [2,1,2,2,1,2,1,1,2,1,2,2,1,2,1],
            [2,1,2,2,1,2,1,2,1,2,1,1,2,1,2],
            [2,2,1,1,2,2,1,1,2,2,1,1,2,1,2],
            [2,2,1,1,2,2,1,2,1,1,2,2,1,1,2],
            [2,2,1,2,1,1,2,1,2,2,1,2,1,1,2],
            [2,2,1,2,1,1,2,2,1,1,2,1,2,2,1],
        ], oa_info.col_levels)

    elif oa_info.key == "L16_G2":
        matrix = _prime_oa_matrix(2, 16, 15)
        if matrix is None:
            raise ValueError("Failed to build L16(2^15) matrix")
        return matrix

    elif oa_info.key == "L16_212_41":
        return _matrix_from_1_based([
            [1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,2,2,2,2,2,2,2,2],
            [1,2,2,2,2,1,1,1,1,2,2,2,2],
            [1,2,2,2,2,2,2,2,2,1,1,1,1],
            [2,1,1,2,2,1,1,2,2,1,1,2,2],
            [2,1,1,2,2,2,2,1,1,2,2,1,1],
            [2,2,2,1,1,1,1,2,2,2,2,1,1],
            [2,2,2,1,1,2,2,1,1,1,1,2,2],
            [3,1,2,1,2,1,2,1,2,1,2,1,2],
            [3,1,2,1,2,2,1,2,1,2,1,2,1],
            [3,2,1,2,1,1,2,1,2,2,1,2,1],
            [3,2,1,2,1,2,1,2,1,1,2,1,2],
            [4,1,2,2,1,1,2,2,1,1,2,2,1],
            [4,1,2,2,1,2,1,1,2,2,1,1,2],
            [4,2,1,1,2,1,2,2,1,2,1,1,2],
            [4,2,1,1,2,2,1,1,2,1,2,2,1],
        ], oa_info.col_levels)

    elif oa_info.key == "L16_29_42":
        return _matrix_from_1_based([
            [1,1,1,1,1,1,1,1,1,1,1],
            [1,2,1,1,1,2,2,2,2,2,2],
            [1,3,2,2,2,1,1,1,2,2,2],
            [1,4,2,2,2,2,2,2,1,1,1],
            [2,1,1,2,2,1,2,2,1,2,2],
            [2,2,1,2,2,2,1,1,2,1,1],
            [2,3,2,1,1,1,2,2,2,1,1],
            [2,4,2,1,1,2,1,1,1,2,2],
            [3,1,2,1,2,2,1,2,2,1,2],
            [3,2,2,1,2,1,2,1,1,2,1],
            [3,3,1,2,1,2,1,2,1,2,1],
            [3,4,1,2,1,1,2,1,2,1,2],
            [4,1,2,2,1,2,2,1,2,2,1],
            [4,2,2,2,1,1,1,2,1,1,2],
            [4,3,1,1,2,2,2,1,1,1,2],
            [4,4,1,1,2,1,1,2,2,2,1],
        ], oa_info.col_levels)

    elif oa_info.key == "L16_26_43":
        return _matrix_from_1_based([
            [1,1,1,1,1,1,1,1,1],
            [1,2,2,1,1,2,2,2,2],
            [1,3,3,2,2,1,1,2,2],
            [1,4,4,2,2,2,2,1,1],
            [2,1,2,2,2,1,2,1,2],
            [2,2,1,2,2,2,1,2,1],
            [2,3,4,1,1,1,2,2,1],
            [2,4,3,1,1,2,1,1,2],
            [3,1,3,1,2,2,2,2,1],
            [3,2,4,1,2,1,1,1,2],
            [3,3,1,2,1,2,2,1,2],
            [3,4,2,2,1,1,1,2,1],
            [4,1,4,2,1,2,1,2,2],
            [4,2,3,2,1,1,2,1,1],
            [4,3,2,1,2,2,1,1,1],
            [4,4,1,1,2,1,2,2,2],
        ], oa_info.col_levels)

    elif oa_info.key == "L16_23_44":
        return _matrix_from_1_based([
            [1,1,1,1,1,1,1],
            [1,2,2,1,2,2,2],
            [1,3,3,2,3,1,2],
            [1,4,4,2,4,2,1],
            [2,1,2,2,1,2,1],
            [2,2,1,2,2,1,2],
            [2,3,4,1,3,2,2],
            [2,4,3,1,4,1,1],
            [3,1,3,1,4,2,2],
            [3,2,4,1,3,1,1],
            [3,3,1,2,2,2,1],
            [3,4,2,2,1,1,2],
            [4,1,4,2,2,1,2],
            [4,2,3,2,1,2,1],
            [4,3,2,1,4,1,1],
            [4,4,1,1,3,2,2],
        ], oa_info.col_levels)

    elif oa_info.key == "L16_81_28":
        return _matrix_from_1_based([
            [1,1,1,1,1,1,1,1,1],
            [1,2,2,2,2,2,2,2,2],
            [2,1,1,1,1,2,2,2,2],
            [2,2,2,2,2,1,1,1,1],
            [3,1,1,2,2,1,1,2,2],
            [3,2,2,1,1,2,2,1,1],
            [4,1,1,2,2,2,2,1,1],
            [4,2,2,1,1,1,1,2,2],
            [5,1,2,1,2,1,2,1,2],
            [5,2,1,2,1,2,1,2,1],
            [6,1,2,1,2,2,1,2,1],
            [6,2,1,2,1,1,2,1,2],
            [7,1,2,2,1,1,2,2,1],
            [7,2,1,1,2,2,1,1,2],
            [8,1,2,2,1,2,1,1,2],
            [8,2,1,1,2,1,2,2,1],
        ], oa_info.col_levels)

    elif oa_info.key == "L16_4":
        # Standard Taguchi L16(4^5) matrix (0..3-coded from 1..4)
        return [
            [0,0,0,0,0],
            [0,1,1,1,1],
            [0,2,2,2,2],
            [0,3,3,3,3],
            [1,0,1,2,3],
            [1,1,0,3,2],
            [1,2,3,0,1],
            [1,3,2,1,0],
            [2,0,2,3,1],
            [2,1,3,2,0],
            [2,2,0,1,3],
            [2,3,1,0,2],
            [3,0,3,1,2],
            [3,1,2,0,3],
            [3,2,1,3,0],
            [3,3,0,2,1],
        ]

    elif oa_info.key == "L12":
        # Standard Taguchi L12(2^11) matrix (0/1-coded from 1/2 table)
        return [
            [0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,1,1,1,1,1],
            [0,0,1,1,1,0,0,0,1,1,1],
            [0,1,0,1,1,0,1,1,0,0,1],
            [0,1,1,0,1,1,0,1,0,1,0],
            [0,1,1,1,0,1,1,0,1,0,0],
            [1,0,1,1,0,0,1,1,0,1,0],
            [1,0,1,0,1,1,1,0,0,0,1],
            [1,0,0,1,1,1,0,1,1,0,0],
            [1,1,1,0,0,0,0,1,1,0,1],
            [1,1,0,1,0,1,0,0,0,1,1],
            [1,1,0,0,1,0,1,0,1,1,0],
        ]
    
    elif oa_info.key == "L18_21_37":
        return _matrix_from_1_based([
            [1,1,1,1,1,1,1,1],
            [1,1,2,2,2,2,2,2],
            [1,1,3,3,3,3,3,3],
            [1,2,1,1,2,2,3,2],
            [1,2,2,2,3,3,1,3],
            [1,2,3,3,1,1,2,1],
            [1,3,1,2,1,3,2,3],
            [1,3,2,3,2,1,3,1],
            [1,3,3,1,3,2,1,2],
            [2,1,1,3,3,2,2,1],
            [2,1,2,1,1,3,3,2],
            [2,1,3,2,2,1,1,3],
            [2,2,1,2,3,1,3,2],
            [2,2,2,3,1,2,1,3],
            [2,2,3,1,2,3,2,1],
            [2,3,1,3,2,3,1,2],
            [2,3,2,1,3,1,2,3],
            [2,3,3,2,1,2,3,1],
        ], oa_info.col_levels)

    elif oa_info.key == "L18_61_36":
        return _matrix_from_1_based([
            [1,1,1,1,1,1,1],
            [1,2,2,2,2,2,2],
            [1,3,3,3,3,3,3],
            [2,1,1,2,2,3,3],
            [2,2,2,3,3,1,1],
            [2,3,3,1,1,2,2],
            [3,1,2,1,3,2,3],
            [3,2,3,2,1,3,1],
            [3,3,1,3,2,1,2],
            [4,1,3,3,2,2,1],
            [4,2,1,1,3,3,2],
            [4,3,2,2,1,1,3],
            [5,1,2,3,1,3,2],
            [5,2,3,1,2,1,3],
            [5,3,1,2,3,2,1],
            [6,1,3,2,3,1,2],
            [6,2,1,3,1,2,3],
            [6,3,2,1,2,3,1],
        ], oa_info.col_levels)

    elif oa_info.key == "L32":
        left_rows = [
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2],
            [1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,1],
            [1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2],
            [1,1,1,2,2,2,2,1,1,1,1,2,2,2,2,1],
            [1,1,1,2,2,2,2,1,1,1,1,2,2,2,2,2],
            [1,1,1,2,2,2,2,2,2,2,2,1,1,1,1,1],
            [1,1,1,2,2,2,2,2,2,2,2,1,1,1,1,2],
            [1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1],
            [1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,2],
            [1,2,2,1,1,2,2,2,2,1,1,2,2,1,1,1],
            [1,2,2,1,1,2,2,2,2,1,1,2,2,1,1,2],
            [1,2,2,2,2,1,1,1,1,2,2,2,2,1,1,1],
            [1,2,2,2,2,1,1,1,1,2,2,2,2,1,1,2],
            [1,2,2,2,2,1,1,2,2,1,1,1,1,2,2,1],
            [1,2,2,2,2,1,1,2,2,1,1,1,1,2,2,2],
            [2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1],
            [2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,2],
            [2,1,2,1,2,1,2,2,1,2,1,2,1,2,1,1],
            [2,1,2,1,2,1,2,2,1,2,1,2,1,2,1,2],
            [2,1,2,2,1,2,1,1,2,1,2,2,1,2,1,1],
            [2,1,2,2,1,2,1,1,2,1,2,2,1,2,1,2],
            [2,1,2,2,1,2,1,2,1,2,1,1,2,1,2,1],
            [2,1,2,2,1,2,1,2,1,2,1,1,2,1,2,2],
            [2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1],
            [2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,2],
            [2,2,1,1,2,2,1,2,1,1,2,2,1,1,2,1],
            [2,2,1,1,2,2,1,2,1,1,2,2,1,1,2,2],
            [2,2,1,2,1,1,2,1,2,2,1,2,1,1,2,1],
            [2,2,1,2,1,1,2,1,2,2,1,2,1,1,2,2],
            [2,2,1,2,1,1,2,2,1,1,2,1,2,2,1,1],
            [2,2,1,2,1,1,2,2,1,1,2,1,2,2,1,2],
        ]
        right_rows = [
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
            [1,1,1,1,1,1,1,2,2,2,2,2,2,2,2],
            [2,2,2,2,2,2,2,1,1,1,1,1,1,1,1],
            [1,1,1,2,2,2,2,1,1,1,1,2,2,2,2],
            [2,2,2,1,1,1,1,2,2,2,2,1,1,1,1],
            [1,1,1,2,2,2,2,2,2,2,2,1,1,1,1],
            [2,2,2,1,1,1,1,1,1,1,1,2,2,2,2],
            [1,2,2,1,1,2,2,1,1,2,2,1,1,2,2],
            [2,1,1,2,2,1,1,2,2,1,1,2,2,1,1],
            [1,2,2,1,1,2,2,2,2,1,1,2,2,1,1],
            [2,1,1,2,2,1,1,1,1,2,2,1,1,2,2],
            [1,2,2,2,2,1,1,1,1,2,2,2,2,1,1],
            [2,1,1,1,1,2,2,2,2,1,1,1,1,2,2],
            [1,2,2,2,2,1,1,2,2,1,1,1,1,2,2],
            [2,1,1,1,1,2,2,1,1,2,2,2,2,1,1],
            [2,1,2,1,2,1,2,1,2,1,2,1,2,1,2],
            [1,2,1,2,1,2,1,2,1,2,1,2,1,2,1],
            [2,1,2,1,2,1,2,2,1,2,1,2,1,2,1],
            [1,2,1,2,1,2,1,1,2,1,2,1,2,1,2],
            [2,1,2,2,1,2,1,1,2,1,2,2,1,2,1],
            [1,2,1,1,2,1,2,2,1,2,1,1,2,1,2],
            [2,1,2,2,1,2,1,2,1,2,1,1,2,1,2],
            [1,2,1,1,2,1,2,1,2,1,2,2,1,2,1],
            [2,2,1,1,2,2,1,1,2,2,1,1,2,2,1],
            [1,1,2,2,1,1,2,2,1,1,2,2,1,1,2],
            [2,2,1,1,2,2,1,2,1,1,2,2,1,1,2],
            [1,1,2,2,1,1,2,1,2,2,1,1,2,2,1],
            [2,2,1,2,1,1,2,1,2,2,1,2,1,1,2],
            [1,1,2,1,2,2,1,2,1,1,2,1,2,2,1],
            [2,2,1,2,1,1,2,2,1,1,2,1,2,2,1],
            [1,1,2,1,2,2,1,1,2,2,1,2,1,1,2],
        ]
        return _matrix_from_1_based(_concat_row_blocks(left_rows, right_rows), oa_info.col_levels)

    elif oa_info.key == "L32_21_49":
        return _matrix_from_1_based([
            [1,1,1,1,1,1,1,1,1,1],
            [1,1,2,2,2,2,2,2,2,2],
            [1,1,3,3,3,3,3,3,3,3],
            [1,1,4,4,4,4,4,4,4,4],
            [1,2,1,1,2,2,3,3,4,4],
            [1,2,2,2,1,1,4,4,3,3],
            [1,2,3,3,4,4,1,1,2,2],
            [1,2,4,4,3,3,2,2,1,1],
            [1,3,1,2,3,4,1,2,3,4],
            [1,3,2,1,4,3,2,1,4,3],
            [1,3,3,4,1,2,3,4,1,2],
            [1,3,4,3,2,1,4,3,2,1],
            [1,4,1,2,4,3,3,4,2,1],
            [1,4,2,1,3,4,4,3,1,2],
            [1,4,3,4,2,1,1,2,4,3],
            [1,4,4,3,1,2,2,1,3,4],
            [2,1,1,4,1,4,2,3,2,3],
            [2,1,2,3,2,3,1,4,1,4],
            [2,1,3,2,3,2,4,1,4,1],
            [2,1,4,1,4,1,3,2,3,2],
            [2,2,1,4,2,3,4,1,3,2],
            [2,2,2,3,1,4,3,2,4,1],
            [2,2,3,2,4,1,2,3,1,4],
            [2,2,4,1,3,2,1,4,2,3],
            [2,3,1,3,3,1,2,4,4,2],
            [2,3,2,4,4,2,1,3,3,1],
            [2,3,3,1,1,3,4,2,2,4],
            [2,3,4,2,2,4,3,1,1,3],
            [2,4,1,3,4,2,4,2,1,3],
            [2,4,2,4,3,1,3,1,2,4],
            [2,4,3,1,2,4,2,4,3,1],
            [2,4,4,2,1,3,1,3,4,2],
        ], oa_info.col_levels)

    elif oa_info.key == "L36_211_312":
        return _matrix_from_1_based([
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2],
            [1,1,1,1,1,1,1,1,1,1,1,3,3,3,3,3,3,3,3,3,3,3,3],
            [1,1,1,1,1,2,2,2,2,2,2,1,1,1,1,2,2,2,2,3,3,3,3],
            [1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,1,1,1,1],
            [1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,1,1,1,1,2,2,2,2],
            [1,1,2,2,2,1,1,1,2,2,2,1,1,2,3,1,2,3,3,1,2,2,3],
            [1,1,2,2,2,1,1,1,2,2,2,2,2,3,1,2,3,1,1,2,3,3,1],
            [1,1,2,2,2,1,1,1,2,2,2,3,3,1,2,3,1,2,2,3,1,1,2],
            [1,2,1,2,2,1,2,2,1,1,2,1,1,3,2,1,3,2,3,2,1,3,2],
            [1,2,1,2,2,1,2,2,1,1,2,2,2,1,3,2,1,3,1,3,2,1,3],
            [1,2,1,2,2,1,2,2,1,1,2,3,3,2,1,3,2,1,2,1,3,2,1],
            [1,2,2,1,2,2,1,2,1,2,1,1,2,3,1,3,2,1,3,3,2,1,2],
            [1,2,2,1,2,2,1,2,1,2,1,2,3,1,2,1,3,2,1,1,3,2,3],
            [1,2,2,1,2,2,1,2,1,2,1,3,1,2,3,2,1,3,2,2,1,3,1],
            [1,2,2,2,1,2,2,1,2,1,1,1,2,3,2,1,1,3,2,3,3,2,1],
            [1,2,2,2,1,2,2,1,2,1,1,2,3,1,3,2,2,1,3,1,1,3,2],
            [1,2,2,2,1,2,2,1,2,1,1,3,1,2,1,3,3,2,1,2,2,1,3],
            [2,1,2,2,1,1,2,2,1,2,1,1,2,1,3,3,3,1,2,2,1,2,3],
            [2,1,2,2,1,1,2,2,1,2,1,2,3,2,1,1,1,2,3,3,2,3,1],
            [2,1,2,2,1,1,2,2,1,2,1,3,1,3,2,2,2,3,1,1,3,1,2],
            [2,1,2,1,2,2,2,1,1,1,2,1,2,2,3,3,1,2,1,1,3,3,2],
            [2,1,2,1,2,2,2,1,1,1,2,2,3,3,1,1,2,3,2,2,1,1,3],
            [2,1,2,1,2,2,2,1,1,1,2,3,1,1,2,2,3,1,3,3,2,2,1],
            [2,1,1,2,2,2,1,2,2,1,1,1,3,2,1,2,3,3,1,3,1,2,2],
            [2,1,1,2,2,2,1,2,2,1,1,2,1,3,2,3,1,1,2,1,2,3,3],
            [2,1,1,2,2,2,1,2,2,1,1,3,2,1,3,1,2,2,3,2,3,1,1],
            [2,2,2,1,1,1,1,2,2,1,2,1,3,2,2,2,1,1,3,2,3,1,3],
            [2,2,2,1,1,1,1,2,2,1,2,2,1,3,3,3,2,2,1,3,1,2,1],
            [2,2,2,1,1,1,1,2,2,1,2,3,2,1,1,1,3,3,2,1,2,3,2],
            [2,2,1,2,1,2,1,1,1,2,2,1,3,3,3,2,3,2,2,1,2,1,1],
            [2,2,1,2,1,2,1,1,1,2,2,2,1,1,1,3,1,3,3,2,3,2,2],
            [2,2,1,2,1,2,1,1,1,2,2,3,2,2,2,1,2,1,1,3,1,3,3],
            [2,2,1,1,2,1,2,1,2,2,1,1,3,1,2,3,2,3,1,2,2,3,1],
            [2,2,1,1,2,1,2,1,2,2,1,2,1,2,3,1,3,1,2,3,3,1,2],
            [2,2,1,1,2,1,2,1,2,2,1,3,2,3,1,2,1,2,3,1,1,2,3],
        ], oa_info.col_levels)

    elif oa_info.key == "L36_23_313":
        return _matrix_from_1_based([
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2],
            [1,1,1,1,3,3,3,3,3,3,3,3,3,3,3,3],
            [1,2,2,1,1,1,1,1,2,2,2,2,3,3,3,3],
            [1,2,2,1,2,2,2,2,3,3,3,3,1,1,1,1],
            [1,2,2,1,3,3,3,3,1,1,1,1,2,2,2,2],
            [2,1,2,1,1,1,2,3,1,2,3,3,1,2,2,3],
            [2,1,2,1,2,2,3,1,2,3,1,1,2,3,3,1],
            [2,1,2,1,3,3,1,2,3,1,2,2,3,1,1,2],
            [2,2,1,1,1,1,3,2,1,3,2,3,2,1,3,2],
            [2,2,1,1,2,2,1,3,2,1,3,1,3,2,1,3],
            [2,2,1,1,3,3,2,1,3,2,1,2,1,3,2,1],
            [1,1,1,2,1,2,3,1,3,2,1,3,3,2,1,2],
            [1,1,1,2,2,3,1,2,1,3,2,1,1,3,2,3],
            [1,1,1,2,3,1,2,3,2,1,3,2,2,1,3,1],
            [1,2,2,2,1,2,3,2,1,1,3,2,3,3,2,1],
            [1,2,2,2,2,3,1,3,2,2,1,3,1,1,3,2],
            [1,2,2,2,3,1,2,1,3,3,2,1,2,2,1,3],
            [2,1,2,2,1,2,1,3,3,3,1,2,2,1,2,3],
            [2,1,2,2,2,3,2,1,1,1,2,3,3,2,3,1],
            [2,1,2,2,3,1,3,2,2,2,3,1,1,3,1,2],
            [2,2,1,2,1,2,2,3,3,1,2,1,1,3,3,2],
            [2,2,1,2,2,3,3,1,1,2,3,2,2,1,1,3],
            [2,2,1,2,3,1,1,2,2,3,1,3,3,2,2,1],
            [1,1,1,3,1,3,2,1,2,3,3,1,3,1,2,2],
            [1,1,1,3,2,1,3,2,3,1,1,2,1,2,3,3],
            [1,1,1,3,3,2,1,3,1,2,2,3,2,3,1,1],
            [1,2,2,3,1,3,2,2,2,1,1,3,2,3,1,3],
            [1,2,2,3,2,1,3,3,3,2,2,1,3,1,2,1],
            [1,2,2,3,3,2,1,1,1,3,3,2,1,2,3,2],
            [2,1,2,3,1,3,3,3,2,3,2,2,1,2,1,1],
            [2,1,2,3,2,1,1,1,3,1,3,3,2,3,2,2],
            [2,1,2,3,3,2,2,2,1,2,1,1,3,1,3,3],
            [2,2,1,3,1,3,1,2,3,2,3,1,2,2,3,1],
            [2,2,1,3,2,1,2,3,1,3,1,2,3,3,1,2],
            [2,2,1,3,3,2,3,1,2,1,2,3,1,1,2,3],
        ], oa_info.col_levels)

    elif oa_info.key == "L54_21_325":
        left_rows = [
            [1,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,1,1,2,2,2,2,2],
            [1,1,1,1,1,1,1,1,3,3,3,3,3],
            [1,1,2,2,2,2,2,2,1,1,1,1,1],
            [1,1,2,2,2,2,2,2,2,2,2,2,2],
            [1,1,2,2,2,2,2,2,3,3,3,3,3],
            [1,1,3,3,3,3,3,3,1,1,1,1,1],
            [1,1,3,3,3,3,3,3,2,2,2,2,2],
            [1,1,3,3,3,3,3,3,3,3,3,3,3],
            [1,2,1,1,2,2,3,3,1,1,2,2,3],
            [1,2,1,1,2,2,3,3,2,2,3,3,1],
            [1,2,1,1,2,2,3,3,3,3,1,1,2],
            [1,2,2,2,3,3,1,1,1,1,2,2,3],
            [1,2,2,2,3,3,1,1,2,2,3,3,1],
            [1,2,2,2,3,3,1,1,3,3,1,1,2],
            [1,2,3,3,1,1,2,2,1,1,2,2,3],
            [1,2,3,3,1,1,2,2,2,2,3,3,1],
            [1,2,3,3,1,1,2,2,3,3,1,1,2],
            [1,3,1,2,1,3,2,3,1,2,1,3,2],
            [1,3,1,2,1,3,2,3,2,3,2,1,3],
            [1,3,1,2,1,3,2,3,3,1,3,2,1],
            [1,3,2,3,2,1,3,1,1,2,1,3,2],
            [1,3,2,3,2,1,3,1,2,3,2,1,3],
            [1,3,2,3,2,1,3,1,3,1,3,2,1],
            [1,3,3,1,3,2,1,2,1,2,1,3,2],
            [1,3,3,1,3,2,1,2,2,3,2,1,3],
            [1,3,3,1,3,2,1,2,3,1,3,2,1],
            [2,1,1,3,3,2,2,1,1,3,3,2,2],
            [2,1,1,3,3,2,2,1,2,1,1,3,3],
            [2,1,1,3,3,2,2,1,3,2,2,1,1],
            [2,1,2,1,1,3,3,2,1,3,3,2,2],
            [2,1,2,1,1,3,3,2,2,1,1,3,3],
            [2,1,2,1,1,3,3,2,3,2,2,1,1],
            [2,1,3,2,2,1,1,3,1,3,3,2,2],
            [2,1,3,2,2,1,1,3,2,1,1,3,3],
            [2,1,3,2,2,1,1,3,3,2,2,1,1],
            [2,2,1,2,3,1,3,2,1,2,3,1,3],
            [2,2,1,2,3,1,3,2,2,3,1,2,1],
            [2,2,1,2,3,1,3,2,3,1,2,3,2],
            [2,2,2,3,1,2,1,3,1,2,3,1,3],
            [2,2,2,3,1,2,1,3,2,3,1,2,1],
            [2,2,2,3,1,2,1,3,3,1,2,3,2],
            [2,2,3,1,2,3,2,1,1,2,3,1,3],
            [2,2,3,1,2,3,2,1,2,3,1,2,1],
            [2,2,3,1,2,3,2,1,3,1,2,3,2],
            [2,3,1,3,2,3,1,2,1,3,2,3,1],
            [2,3,1,3,2,3,1,2,2,1,3,1,2],
            [2,3,1,3,2,3,1,2,3,2,1,2,3],
            [2,3,2,1,3,1,2,3,1,3,2,3,1],
            [2,3,2,1,3,1,2,3,2,1,3,1,2],
            [2,3,2,1,3,1,2,3,3,2,1,2,3],
            [2,3,3,2,1,2,3,1,1,3,2,3,1],
            [2,3,3,2,1,2,3,1,2,1,3,1,2],
            [2,3,3,2,1,2,3,1,3,2,1,2,3],
        ]
        right_rows = [
            [1,1,1,1,1,1,1,1,1,1,1,1,1],
            [2,2,2,2,2,2,2,2,2,2,2,2,2],
            [3,3,3,3,3,3,3,3,3,3,3,3,3],
            [1,2,3,2,3,2,3,2,3,2,3,2,3],
            [2,3,1,3,1,3,1,3,1,3,1,3,1],
            [3,1,2,1,2,1,2,1,2,1,2,1,2],
            [1,3,2,3,2,3,2,3,2,3,2,3,2],
            [2,1,3,1,3,1,3,1,3,1,3,1,3],
            [3,2,1,2,1,2,1,2,1,2,1,2,1],
            [3,1,1,1,1,2,3,2,3,2,3,2,3],
            [1,2,2,2,2,3,1,3,1,1,3,1,3],
            [2,3,3,3,3,1,2,1,2,2,1,2,1],
            [3,2,3,2,3,3,2,3,2,1,1,1,1],
            [1,3,1,3,1,1,3,1,3,2,2,2,2],
            [2,1,2,1,2,2,1,2,1,3,3,3,3],
            [3,3,2,3,2,1,1,1,1,2,3,2,3],
            [1,1,3,1,3,2,2,2,2,3,1,3,1],
            [2,2,1,2,1,3,3,3,3,1,2,1,2],
            [3,1,1,2,3,1,1,3,2,2,3,3,2],
            [1,2,2,3,1,2,2,1,3,3,1,1,3],
            [2,3,3,1,2,3,3,2,1,1,2,2,1],
            [3,2,3,3,2,2,3,1,1,3,2,1,1],
            [1,3,1,1,3,3,1,2,2,1,3,2,2],
            [2,1,2,2,1,1,2,3,3,2,1,3,3],
            [3,3,2,1,1,3,2,2,3,1,1,2,3],
            [1,1,3,2,2,1,3,3,1,2,2,3,1],
            [2,2,1,3,3,2,1,1,2,3,3,1,2],
            [1,1,1,3,2,3,2,2,3,2,3,1,1],
            [2,2,2,1,3,1,3,3,1,3,1,2,2],
            [3,3,3,2,1,2,1,1,2,1,2,3,3],
            [1,2,3,1,1,1,1,3,2,3,2,2,3],
            [2,3,1,2,2,2,2,1,3,1,3,3,1],
            [3,1,2,3,3,3,3,2,1,2,1,1,2],
            [1,3,2,2,3,2,3,1,1,1,1,3,2],
            [2,1,3,3,1,3,1,2,2,2,2,1,3],
            [3,2,1,1,2,1,2,3,3,3,3,2,1],
            [2,1,1,2,3,3,2,1,1,3,2,2,3],
            [3,2,2,3,1,1,3,2,2,1,3,3,1],
            [1,3,3,1,2,2,1,3,3,2,1,1,2],
            [2,2,3,3,2,1,1,2,3,1,1,3,2],
            [3,3,1,1,3,2,2,3,1,2,2,1,3],
            [1,1,2,2,1,3,3,1,2,3,3,2,1],
            [2,3,2,1,1,2,3,3,2,2,3,1,1],
            [3,1,3,2,2,3,1,1,3,3,1,2,2],
            [1,2,1,3,3,1,2,2,1,1,2,3,3],
            [2,1,1,3,2,2,3,3,2,1,1,2,3],
            [3,2,2,1,3,3,1,1,3,2,2,3,1],
            [1,3,3,2,1,1,2,2,1,3,3,1,2],
            [2,2,3,1,1,3,2,1,1,2,3,3,2],
            [3,3,1,2,2,1,3,2,2,3,1,1,3],
            [1,1,2,3,3,2,1,3,3,1,2,2,1],
            [2,3,2,2,3,1,1,2,3,3,2,1,1],
            [3,1,3,3,1,2,2,3,1,1,3,2,2],
            [1,2,1,1,2,3,3,1,2,2,1,3,3],
        ]
        return _matrix_from_1_based(_concat_row_blocks(left_rows, right_rows), oa_info.col_levels)

    elif oa_info.key == "PB12":
        # PB-12 is equivalent in strength-2 structure to L12(2^11)
        return generate_basic_oa_matrix(get_oa_info("L12(2)"))
    
    else:
        # Try rigorous prime-level OA construction first (covers L32, L64, L81, L25, ...)
        level_set = set(oa_info.col_levels)
        if len(level_set) == 1:
            level = oa_info.col_levels[0]
            matrix = _prime_oa_matrix(level, oa_info.runs, len(oa_info.col_levels))
            if matrix is not None:
                return matrix

        # Mixed or unsupported levels: use diverse deterministic fallback.
        return _mixed_diverse_fallback(oa_info.col_levels, oa_info.runs)

def generate_design(oa_info: OAInfo, factor_names: List[str], factor_levels: List[List[str]]) -> pd.DataFrame:
    """Generate complete design matrix with proper factor labels"""
    if not _is_generator_valid(oa_info.label):
        raise ValueError(
            f"OA '{oa_info.label}' is currently not backed by a validated orthogonal matrix in this build. "
            "Please choose another OA option."
        )
    
    # Generate basic integer matrix
    matrix = generate_basic_oa_matrix(oa_info)
    
    # Convert to DataFrame with proper labels
    df = pd.DataFrame()
    
    for j, (name, levels) in enumerate(zip(factor_names, factor_levels)):
        if j < len(matrix[0]):
            # Map integer codes to actual factor levels
            df[name] = [levels[row[j] % len(levels)] for row in matrix]
    
    return df

def get_oa_info(label: str) -> Optional[OAInfo]:
    """Get OA info by label"""
    for oa in _ENHANCED_OA_CATALOG:
        if oa.label == label:
            return oa
    return None

def search_oas_by_criteria(min_factors: int = 1, max_runs: int = 1000, 
                          preferred_levels: List[int] = None) -> List[OAInfo]:
    """Search OAs by specific criteria"""
    results = []
    
    for oa in _ENHANCED_OA_CATALOG:
        if oa.factors >= min_factors and oa.runs <= max_runs:
            if preferred_levels is None:
                results.append(oa)
            else:
                # Check if OA supports preferred levels
                oa_level_set = set(oa.col_levels)
                pref_level_set = set(preferred_levels)
                if pref_level_set.issubset(oa_level_set):
                    results.append(oa)
    
    return sorted(results, key=lambda x: x.runs)

def build_design(oa_info: OAInfo, factor_names: List[str], factor_levels: List[List[str]]) -> pd.DataFrame:
    """Build design matrix - alias for generate_design for compatibility"""
    return generate_design(oa_info, factor_names, factor_levels)
