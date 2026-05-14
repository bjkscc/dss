"""Per-atom element property features fetched from pymatgen at runtime."""

import warnings

import torch
from pymatgen.core.periodic_table import Element

warnings.filterwarnings("ignore", message="No Pauling electronegativity")

# Five core features: electronegativity, atomic radius, group, period, atomic mass
FEATURE_NAMES = ["electronegativity", "atomic_radius", "group", "period", "atomic_mass"]
N_FEATURES = len(FEATURE_NAMES)
MAX_Z = 119  # atomic numbers 0..118 (0 = padding)


def _safe_float(value, default=0.0):
    """Convert pymatgen FloatWithUnit or None to plain float."""
    if value is None:
        return default
    try:
        result = float(value)
        if result != result:  # NaN check
            return default
        return result
    except (TypeError, ValueError):
        return default


def build_element_property_table() -> torch.Tensor:
    """Build normalized element property table of shape (MAX_Z, N_FEATURES).

    Properties are z-score normalized per column across Z=1..118.
    Row 0 is all zeros (padding for atomic number 0).
    """
    raw = torch.zeros(MAX_Z, N_FEATURES)

    for z in range(1, MAX_Z):
        try:
            e = Element.from_Z(z)
            raw[z, 0] = _safe_float(e.X)                        # electronegativity
            raw[z, 1] = _safe_float(e.atomic_radius)             # atomic radius (Ang)
            raw[z, 2] = float(e.group if e.group is not None else 0)   # group
            raw[z, 3] = float(e.row if e.row is not None else 0)       # period
            raw[z, 4] = _safe_float(e.atomic_mass)               # atomic mass (amu)
        except Exception:
            pass  # skip elements that fail (e.g., beyond the periodic table)

    # Z-score normalize per column over non-zero rows
    valid = raw[1:].clone()  # exclude row 0 (padding)
    mean = valid.mean(dim=0)
    std = valid.std(dim=0, unbiased=False)
    std[std < 1e-8] = 1.0  # avoid division by zero

    table = (raw - mean) / std
    table[0] = 0.0  # padding row stays zero
    return table


def get_element_features(
    atomic_numbers: torch.Tensor, table: torch.Tensor
) -> torch.Tensor:
    """Look up element features by atomic number.

    Args:
        atomic_numbers: (N,) long tensor of atomic numbers
        table: (MAX_Z, N_FEATURES) element property table

    Returns:
        (N, N_FEATURES) feature tensor
    """
    z = atomic_numbers.long().clamp(0, MAX_Z - 1)
    return table[z]
