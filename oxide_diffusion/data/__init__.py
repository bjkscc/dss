"""Data loading and processing for oxide-supported metal cluster systems."""

from .arc_parser import (
    OXIDE_SUPPORT_ELEMENTS,
    SUPPORTED_METALS,
    auto_detect_oxide_type,
    frames_to_ase_atoms,
    get_support_elements,
    parse_arc_file,
    split_by_trajectory,
)
from .dataset import create_oxide_dataset, create_trajectory_split
from .element_properties import (
    FEATURE_NAMES,
    N_FEATURES,
    build_element_property_table,
    get_element_features,
)
