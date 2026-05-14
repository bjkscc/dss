"""BIOSYM archive 2 (.arc) file parser for oxide-supported metal cluster data."""

import os
import re
from typing import Dict, List, Set, Tuple

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

# ── Oxide support registry ──────────────────────────────────────────

OXIDE_SUPPORT_ELEMENTS: Dict[str, Set[str]] = {
    "Al2O3": {"Al", "O"},
    "SiO2": {"Si", "O"},
    "TiO2": {"Ti", "O"},
    "ZnO": {"Zn", "O"},
    "CeO2": {"Ce", "O"},
    "ZrO2": {"Zr", "O"},
    "Y2O3": {"Y", "O"},
    "Sc2O3": {"Sc", "O"},
    "MgO": {"Mg", "O"},
    "Li2O": {"Li", "O"},
    "MnO2": {"Mn", "O"},
    "Cr2O3": {"Cr", "O"},
    "V2O5": {"V", "O"},
    "HfO2": {"Hf", "O"},
    "GeO2": {"Ge", "O"},
    "Bi2O3": {"Bi", "O"},
    "SnO2": {"Sn", "O"},
    "Co3O4": {"Co", "O"},
    "La2O3": {"La", "O"},
}

SUPPORTED_METALS: Set[str] = {
    "Pt", "Rh", "Ru", "Ag", "Au", "Cu", "Pd", "Ni",
}


def get_support_elements(oxide_type: str) -> Set[str]:
    """Return the set of element symbols that constitute the oxide support."""
    if oxide_type not in OXIDE_SUPPORT_ELEMENTS:
        raise ValueError(
            f"Unknown oxide type: {oxide_type}. "
            f"Known types: {list(OXIDE_SUPPORT_ELEMENTS.keys())}"
        )
    return OXIDE_SUPPORT_ELEMENTS[oxide_type]


def auto_detect_oxide_type(filename: str) -> str:
    """Extract oxide type from .arc filename, e.g. 'AuNiCuPdPt-ZnO.arc' -> 'ZnO'."""
    basename = os.path.splitext(os.path.basename(filename))[0]
    # Split on last hyphen: everything after it is the oxide type
    if "-" in basename:
        return basename.rsplit("-", 1)[-1]
    raise ValueError(
        f"Cannot auto-detect oxide type from filename: {filename}. "
        f"Expected format like 'Metals-OxideType.arc'"
    )


# ── Parser ───────────────────────────────────────────────────────────

def parse_arc_file(filepath: str) -> List[Dict]:
    """Parse a BIOSYM archive 2 file into a list of frame dicts.

    Each frame dict contains:
        trajectory_id: int  — unique trajectory index
        step: int           — MD step within trajectory
        energy: float       — total potential energy (eV)
        metric: float       — MD metric (temperature or time)
        symmetry: str       — space group label (usually "C1")
        cell: np.ndarray    — (6,) array [a, b, c, alpha, beta, gamma]
        symbols: List[str]  — element symbols for each atom
        positions: np.ndarray — (N, 3) atomic coordinates
    """
    frames = []
    current_frame = None
    current_atoms_symbols = []
    current_atoms_positions = []
    current_traj_id = 0
    prev_step = None
    in_atoms = False

    with open(filepath, "r") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")

            if not line or line.startswith("!DATE"):
                continue

            if line.startswith("!BIOSYM") or line.startswith("PBC=ON"):
                continue

            if line.startswith("end"):
                if in_atoms and current_frame is not None:
                    current_frame["symbols"] = current_atoms_symbols
                    current_frame["positions"] = np.array(current_atoms_positions)
                    frames.append(current_frame)
                    current_frame = None
                    current_atoms_symbols = []
                    current_atoms_positions = []
                    in_atoms = False
                continue

            if line.lstrip().startswith("Energy"):
                # Energy line: "                      Energy   step   metric   energy   symmetry"
                parts = line.split()
                if len(parts) < 5:
                    continue
                step = int(parts[1])
                metric = float(parts[2])
                energy = float(parts[3])
                symmetry = parts[4]

                # Detect new trajectory: step resets to 0 or decreases
                if prev_step is not None and step <= prev_step:
                    current_traj_id += 1
                prev_step = step

                current_frame = {
                    "trajectory_id": current_traj_id,
                    "step": step,
                    "energy": energy,
                    "metric": metric,
                    "symmetry": symmetry,
                    "cell": None,
                }
                in_atoms = False
                continue

            if line.startswith("PBC"):
                # PBC line: "PBC   a   b   c   alpha   beta   gamma"
                parts = line.split()
                cell = np.array([float(x) for x in parts[1:7]])
                if current_frame is not None:
                    current_frame["cell"] = cell
                in_atoms = True
                continue

            if in_atoms:
                # Atom line: "element x y z CORE serial element element 0.0000 index"
                parts = line.split()
                if len(parts) >= 4:
                    symbol = parts[0]
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    current_atoms_symbols.append(symbol)
                    current_atoms_positions.append([x, y, z])

    return frames


# ── Conversion ────────────────────────────────────────────────────────

def frames_to_ase_atoms(frames: List[Dict]) -> List[Atoms]:
    """Convert parsed frame dicts to ASE Atoms objects with energy attached."""
    atoms_list = []
    for frame in frames:
        a = Atoms(
            symbols=frame["symbols"],
            positions=frame["positions"],
            cell=frame["cell"][:3],  # a, b, c
            pbc=True,
        )
        a.set_calculator(
            SinglePointCalculator(a, energy=frame["energy"])
        )
        atoms_list.append(a)
    return atoms_list


# ── Trajectory grouping ───────────────────────────────────────────────

def split_by_trajectory(frames: List[Dict]) -> List[List[Dict]]:
    """Group frames by trajectory_id for trajectory-level train/val split."""
    trajectories: Dict[int, List[Dict]] = {}
    for frame in frames:
        tid = frame["trajectory_id"]
        trajectories.setdefault(tid, []).append(frame)
    return list(trajectories.values())
