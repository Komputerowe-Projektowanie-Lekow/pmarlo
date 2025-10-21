"""
CV Bias Potential for Enhanced Sampling in OpenMM.

This module implements a wrapper that transforms Deep-TICA collective variable (CV)
outputs into bias potentials suitable for enhanced sampling. The wrapper now embeds
TorchScript-based feature extraction so that OpenMM's TorchForce only needs to pass
atomic positions (and box vectors) at every MD step.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = ["CVBiasPotential", "HarmonicExpansionBias", "create_cv_bias_potential"]


class HarmonicExpansionBias(nn.Module):
    """
    Harmonic restraint in collective variable space.

    Applies a quadratic penalty proportional to squared CV values: E_bias = k·Σ(cv_i²).
    This is a restraint, not an exploration bias—it encourages non-zero CV values but
    does not guarantee better conformational sampling.

    See example_programs/app_usecase/app/CV_REQUIREMENTS.md for physics discussion.

    Parameters
    ----------
    strength : float
        Bias strength (force constant k) in kJ/mol. Default 10.0 is a starting point;
        tune based on your system (see CV_INTEGRATION_GUIDE.md).
    """

    def __init__(self, strength: float = 10.0) -> None:
        super().__init__()
        self.register_buffer(
            "strength", torch.tensor(float(strength), dtype=torch.float32)
        )

    def forward(self, cvs: Tensor) -> Tensor:
        """Compute harmonic expansion bias energy."""
        energy = self.strength * torch.sum(cvs * cvs, dim=-1)
        return energy


class CVBiasPotential(nn.Module):
    """
    Wrap a Deep-TICA CV model with TorchScript feature extraction and biasing.

    Architecture (all TorchScript, no Python per-step):
        1. Extract molecular features directly from atomic positions (distances, angles, dihedrals)
        2. Scale features using fitted StandardScaler parameters
        3. Evaluate the Deep-TICA model to obtain CVs
        4. Apply harmonic restraint: E = k·Σ(cv_i²)
        5. Return energy in kJ/mol (OpenMM computes forces via automatic differentiation)

    See example_programs/app_usecase/app/CV_INTEGRATION_GUIDE.md for usage guide.
    """

    def __init__(
        self,
        cv_model: nn.Module,
        scaler_mean: np.ndarray | Tensor,
        scaler_scale: np.ndarray | Tensor,
        *,
        feature_extractor: nn.Module,
        feature_spec_hash: str,
        bias_type: str = "harmonic_expansion",
        bias_strength: float = 10.0,
        feature_names: Optional[list[str]] = None,
    ) -> None:
        super().__init__()

        if not isinstance(cv_model, nn.Module):
            raise TypeError("cv_model must be a torch.nn.Module instance")
        if not isinstance(feature_extractor, nn.Module):
            raise TypeError("feature_extractor must be a torch.nn.Module instance")

        self.cv_model = cv_model
        self.feature_extractor = feature_extractor
        self.feature_spec_sha256: str = str(feature_spec_hash)
        self.bias_type = str(bias_type)
        self.bias_strength = float(bias_strength)
        self.feature_names = list(feature_names or [])

        self.register_buffer(
            "scaler_mean",
            torch.as_tensor(np.asarray(scaler_mean, dtype=np.float32)),
        )
        self.register_buffer(
            "scaler_scale",
            torch.as_tensor(np.asarray(scaler_scale, dtype=np.float32)),
        )
        hash_tensor = torch.tensor(
            list(str(feature_spec_hash).encode("ascii")), dtype=torch.uint8
        )
        self.register_buffer("_feature_spec_hash_bytes", hash_tensor, persistent=True)

        if torch.any(self.scaler_scale == 0):
            raise ValueError("scaler_scale contains zero; cannot normalise features")
        if self.bias_type != "harmonic_expansion":
            raise ValueError(f"unsupported bias type '{bias_type}'")

        self.bias_potential = HarmonicExpansionBias(strength=self.bias_strength)
        self.cv_model.eval()

        self.feature_count = int(self.scaler_mean.numel())
        self.atom_count = int(
            getattr(feature_extractor, "atom_count", self.feature_count)
        )
        self.uses_periodic_boundary_conditions: bool = bool(
            getattr(feature_extractor, "use_pbc", True)
        )

        logger.info(
            "Created CVBiasPotential (%s) with %d features and bias strength %.2f kJ/mol",
            self.bias_type,
            self.feature_count,
            self.bias_strength,
        )

    def forward(self, positions: Tensor, box: Tensor) -> Tensor:
        if positions.dim() != 2 or positions.size(-1) != 3:
            raise RuntimeError("positions tensor must have shape (N, 3)")
        if positions.size(0) < self.atom_count:
            raise RuntimeError(
                f"expected at least {self.atom_count} atoms, received {positions.size(0)}"
            )
        if box.dim() != 2 or box.size(0) != 3 or box.size(1) != 3:
            raise RuntimeError("box tensor must have shape (3, 3)")

        features = self.feature_extractor(positions, box)
        if features.dim() == 1:
            features = features.unsqueeze(0)

        if features.size(-1) != self.feature_count:
            raise RuntimeError(
                f"feature extractor returned {features.size(-1)} features "
                f"but scaler expects {self.feature_count}"
            )

        scaled = (features - self.scaler_mean) / self.scaler_scale
        cvs = self.cv_model(scaled)
        energy = self.bias_potential(cvs)

        if energy.dim() == 0:
            return energy
        if energy.dim() == 1 and energy.numel() == 1:
            return energy.squeeze(0)
        if energy.dim() == 1:
            return energy
        raise RuntimeError("bias potential returned unexpected tensor shape")

    @torch.jit.export
    def compute_cvs(self, positions: Tensor, box: Tensor) -> Tensor:
        if positions.dim() != 2 or positions.size(-1) != 3:
            raise RuntimeError("positions tensor must have shape (N, 3)")
        features = self.feature_extractor(positions, box)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        scaled = (features - self.scaler_mean) / self.scaler_scale
        return self.cv_model(scaled)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "bias_type": self.bias_type,
            "bias_strength": self.bias_strength,
            "feature_names": list(self.feature_names),
            "n_features": int(self.feature_count),
            "cv_model_type": self.cv_model.__class__.__name__,
            "feature_spec_sha256": self.feature_spec_sha256,
            "uses_pbc": self.uses_periodic_boundary_conditions,
            "atom_count": int(self.atom_count),
        }

    @torch.jit.export
    def feature_spec_hash_bytes(self) -> Tensor:
        return self._feature_spec_hash_bytes


def create_cv_bias_potential(
    cv_model: nn.Module,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    *,
    feature_extractor: nn.Module,
    feature_spec_hash: str,
    bias_strength: float = 10.0,
    feature_names: Optional[list[str]] = None,
) -> CVBiasPotential:
    return CVBiasPotential(
        cv_model=cv_model,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        feature_extractor=feature_extractor,
        feature_spec_hash=feature_spec_hash,
        bias_type="harmonic_expansion",
        bias_strength=bias_strength,
        feature_names=feature_names,
    )
