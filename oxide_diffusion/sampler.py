"""Sampling for oxide-supported metal cluster generation.

Adapted from dss/helpers.py:sample(). Key differences:
- Species-based mask (oxide support fixed, metal cluster diffuses)
- No z_confinement (full 3D diffusion)
- Uses diffusion.sample() (Euler-Maruyama), NOT regressor_guidance_sample()
  since there is no potential model
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import schnetpack as spk
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from .data.arc_parser import get_support_elements


def sample_clusters(
    diffusion,
    num_samples: int,
    template: Atoms,
    oxide_type: str,
    num_steps: int = 1000,
    seed: Optional[int] = None,
) -> List[Atoms]:
    """Generate metal cluster structures on oxide support via diffusion sampling.

    Args:
        diffusion: trained VPDiffusion module (score-only, no potential model)
        num_samples: number of structures to generate
        template: ASE Atoms — oxide support slab with fixed positions.
                  Metal atoms to be added are specified by the template's
                  non-support atoms (auto-detected from oxide_type).
        oxide_type: oxide type string (e.g. "ZnO"), used to determine
                    which atoms are support vs. metal cluster
        num_steps: Euler-Maruyama integration steps
        seed: random seed for reproducibility

    Returns:
        List of ASE Atoms with generated metal cluster positions
    """
    if seed is not None:
        torch.manual_seed(seed)

    support_elements = get_support_elements(oxide_type)
    template_symbols = template.get_chemical_symbols()
    is_support = [s in support_elements for s in template_symbols]

    # Diffusing atoms are those NOT in the support
    metal_indices = [i for i, s in enumerate(template_symbols) if not is_support[i]]
    metal_symbols = [template_symbols[i] for i in metal_indices]

    if len(metal_symbols) == 0:
        raise ValueError(
            "Template has no metal cluster atoms. "
            f"Template elements: {set(template_symbols)}. "
            f"Support elements ({oxide_type}): {support_elements}"
        )

    print(f"Generating {num_samples} structures")
    print(f"  Oxide: {oxide_type}")
    print(f"  Support atoms: {sum(is_support)}")
    print(f"  Metal cluster atoms: {len(metal_symbols)} ({metal_symbols})")

    # Build mask: support=True (fixed), metal=False (diffusing)
    mask = np.stack([np.array(is_support)] * 3, axis=-1)

    converter = spk.interfaces.AtomsConverter(
        neighbor_list=None,
        additional_inputs={
            "mask": torch.tensor(np.tile(mask, (1, 1)).reshape(-1, 3)),
        },
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    n_split = min(64, num_samples)
    all_atoms = []

    for batch_start in range(0, num_samples, n_split):
        current_batch_size = min(n_split, num_samples - batch_start)

        atoms_data = []
        for _ in range(current_batch_size):
            # Template atoms with same composition; metal cluster positions
            # are randomized before diffusion inside VPDiffusion
            atoms_data.append(template.copy())

        data = converter(atoms_data)
        # Fix _pbc shape (same hack as original dss)
        data["_pbc"] = data["_pbc"].view(-1)

        # Plain Euler-Maruyama sampling (no potential guidance)
        batch = diffusion.sample(data, num_steps=num_steps, save_traj=False)

        # Split batch back to individual structures
        batch_list = diffusion._split_batch(batch, keep_ef=True)
        for b in batch_list:
            a = Atoms(
                numbers=b["_atomic_numbers"].cpu().detach().numpy(),
                positions=b["_positions"].cpu().detach().numpy(),
                cell=b["_cell"].cpu().detach().numpy().reshape(3, 3),
                pbc=b["_pbc"].cpu().detach().numpy(),
            )
            all_atoms.append(a)

    # Verify oxide support atoms stayed fixed
    support_positions_orig = template.get_positions()[np.array(is_support)]
    support_positions_gen = all_atoms[0].get_positions()[
        [i for i, s in enumerate(all_atoms[0].get_chemical_symbols()) if s in support_elements]
    ]
    max_drift = np.abs(support_positions_orig - support_positions_gen).max()
    if max_drift > 1e-4:
        print(f"Warning: support atoms drifted by {max_drift:.2e} Ang (should be 0)")

    return all_atoms


def make_template(
    slab: Atoms, metal_symbols: List[str]
) -> Atoms:
    """Create a template from an oxide slab and a list of metal symbols.

    Metal atoms are added at random positions within the unit cell.
    During diffusion, these positions will be randomized and then denoised.

    Args:
        slab: ASE Atoms — the oxide support slab
        metal_symbols: list of element symbols for the metal cluster
                       (e.g., ["Pt", "Pt", "Pt", "Pt"] for Pt4)

    Returns:
        Combined ASE Atoms template with slab + uninitialized metal atoms
    """
    all_symbols = list(slab.get_chemical_symbols()) + metal_symbols
    n_slab = len(slab)
    n_metal = len(metal_symbols)

    # Place metal atoms at random positions within the cell
    rng = np.random.default_rng()
    cell = slab.get_cell()[:]
    fractional = rng.random((n_metal, 3))
    metal_positions = fractional @ cell

    all_positions = np.vstack([slab.get_positions(), metal_positions])

    template = Atoms(
        symbols=all_symbols,
        positions=all_positions,
        cell=slab.get_cell(),
        pbc=slab.get_pbc(),
    )
    return template
