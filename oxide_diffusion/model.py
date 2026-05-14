"""ClusterScoreModel with per-atom element property features."""

from typing import Dict, Optional

import schnetpack as spk
import torch
from schnetpack import properties

from dss.diffusion import VPDiffusion
from dss.models.conditionings import Conditioning
from dss.models.score import ConditionedScoreModel, build_gated_equivariant_mlp
from dss.utils import TorchNeighborList

from .data.element_properties import build_element_property_table, N_FEATURES


class ClusterScoreModel(ConditionedScoreModel):
    """ConditionedScoreModel extended with per-atom element property features.

    Element features (electronegativity, atomic radius, group, period, atomic mass)
    are concatenated to the scalar representation after time embedding and conditioning.
    """

    def __init__(
        self,
        representation,
        element_features: torch.Tensor,
        time_dim: int = 2,
        conditioning=None,
        gated_blocks: int = 3,
        **kwargs,
    ):
        super().__init__(
            representation=representation,
            time_dim=time_dim,
            conditioning=conditioning,
            gated_blocks=gated_blocks,
            **kwargs,
        )
        # Register element feature lookup table as a persistent buffer
        self.register_buffer("element_features", element_features)
        self.element_feat_dim = element_features.shape[-1]

        # Rebuild the gated MLP with expanded scalar input dimension
        total_scalar_dim = (
            self.representation.n_atom_basis
            + time_dim
            + self.cond_dim
            + self.element_feat_dim
        )
        self.net = build_gated_equivariant_mlp(
            total_scalar_dim,
            self.representation.n_atom_basis,
            1,
            n_layers=gated_blocks,
        )

    def forward(
        self,
        batch: Dict,
        t: Optional[torch.Tensor] = None,
        prob: float = 0.0,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get representations from PaiNN
        if (
            "scalar_representation" not in batch
            and "vector_representation" not in batch
        ):
            inputs = self.representation(batch)
        else:
            inputs = batch

        scalar_representation = inputs["scalar_representation"]
        vector_representation = inputs["vector_representation"]

        # Time embedding
        if t is None:
            time_cond = torch.zeros(
                (scalar_representation.shape[0], self.time_dim),
                device=self.device,
            )
        else:
            time_cond = torch.cat(
                (torch.sin(self.omega * t), torch.cos(self.omega * t)), dim=-1
            )

        scalar_representation = torch.cat((scalar_representation, time_cond), dim=-1)

        # Conditioning (e.g., energy)
        if self.conditioning is not None:
            cond = self.conditioning(batch, prob=prob, condition=condition)
            scalar_representation = torch.cat((scalar_representation, cond), dim=-1)

        # Per-atom element features
        z = inputs[properties.Z]
        elem_feat = self.element_features[z.long()]
        scalar_representation = torch.cat((scalar_representation, elem_feat), dim=-1)

        scalar, vector = self.net([scalar_representation, vector_representation])
        return vector


def get_oxide_diffusion_model(
    cutoff: float = 6.0,
    n_atom_basis: int = 64,
    n_rbf: int = 30,
    n_interactions: int = 4,
    gated_blocks: int = 4,
    beta_max: float = 3.0,
    beta_min: float = 1e-2,
    lr: float = 1e-3,
    neighbour_list=None,
    condition_config: Optional[dict] = None,
):
    """Build the score-only diffusion model for oxide-supported metal clusters.

    Differences from dss.helpers.get_diffusion_model:
    - Uses ClusterScoreModel with per-atom element features
    - No potential model (score-only training)
    - Energy conditioning retained
    """
    if neighbour_list is None:
        neighbour_list = TorchNeighborList(cutoff)

    if condition_config is None:
        condition_config = {"train_prob": 0.5}

    # Element property table
    element_table = build_element_property_table()

    # PaiNN representation
    radial_basis = spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)
    representation = spk.representation.PaiNN(
        n_atom_basis=n_atom_basis,
        n_interactions=n_interactions,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff),
    )

    # Energy conditioning
    conditioning = Conditioning(dim=2, key="energy", tau=1.0)

    # Score model with element features
    score_model = ClusterScoreModel(
        representation=representation,
        element_features=element_table,
        time_dim=2,
        conditioning=conditioning,
        gated_blocks=gated_blocks,
    )

    # VPDiffusion — score-only, no potential model
    diffusion = VPDiffusion(
        score_model=score_model,
        neighbour_list=neighbour_list,
        potential_model=None,
        beta_max=beta_max,
        beta_min=beta_min,
        condition_config=condition_config,
        optim_config={"lr": lr},
        scheduler_config={"factor": 0.90, "patience": 100},
    )

    return diffusion, neighbour_list
