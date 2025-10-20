from __future__ import annotations

"""TorchScript feature extraction for DeepTICA collective variable models.

This module encapsulates all position -> feature transformations that must run
inside TorchScript when the model is executed by OpenMM's TorchForce.  Feature
extraction here must be completely free of Python-side callbacks and rely on
vectorised tensor operations so that it can execute efficiently on CPU-only
setups.

The exported module currently supports the following feature primitives:

* pairwise distances (minimum-image under periodic boundary conditions)
* bond angles (three-body)
* dihedral / torsion angles (four-body)

Each primitive is described by a specification dictionary and the overall
feature list is ordered as provided by the caller.  The normalisation helpers
defined below guarantee a deterministic layout which is then embedded into a
TorchScript module for execution at MD step time.
"""

from dataclasses import dataclass
from numbers import Real
from typing import Iterable, List, Mapping, MutableMapping, Sequence

import torch
from torch import Tensor, nn

__all__ = [
    "FeatureSpecError",
    "NormalizedFeatureSpec",
    "build_feature_extractor_module",
    "canonicalize_feature_spec",
]


class FeatureSpecError(ValueError):
    """Raised when the provided feature specification is invalid."""


@dataclass(frozen=True)
class NormalizedFeatureSpec:
    """Canonicalised feature specification used to drive TorchScript extraction."""

    feature_names: tuple[str, ...]
    use_pbc: bool
    atom_count: int
    distance_pairs: Tensor
    distance_positions: Tensor
    distance_pbc: Tensor
    angle_triplets: Tensor
    angle_positions: Tensor
    angle_pbc: Tensor
    dihedral_quads: Tensor
    dihedral_positions: Tensor
    dihedral_pbc: Tensor
    feature_weights: Tensor

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def _as_int_sequence(values: Iterable[int], expected: int, *, label: str) -> List[int]:
    items = [int(v) for v in values]
    if len(items) != expected:
        raise FeatureSpecError(f"{label} requires exactly {expected} atom indices")
    if any(i < 0 for i in items):
        raise FeatureSpecError(f"{label} indices must be non-negative")
    return items


@dataclass(frozen=True)
class _FeatureRecord:
    name: str
    feature_type: str
    atoms: List[int]
    weight: float
    pbc: bool
    position: int


def _coerce_weight(raw_weight: object, feature_name: str) -> float:
    if isinstance(raw_weight, Real):
        return float(raw_weight)
    if isinstance(raw_weight, str):
        try:
            return float(raw_weight)
        except ValueError as exc:
            raise FeatureSpecError(
                f"weight for feature '{feature_name}' must be numeric"
            ) from exc
    raise FeatureSpecError(f"weight for feature '{feature_name}' must be numeric")


def _parse_feature_entry(
    entry: Mapping[str, object], position: int, default_pbc: bool
) -> _FeatureRecord:
    if not isinstance(entry, Mapping):
        raise FeatureSpecError("each feature entry must be a mapping")
    entry_map: MutableMapping[str, object] = dict(entry)
    raw_type = entry_map.get("type")
    if raw_type is None:
        raise FeatureSpecError("feature entry missing 'type'")
    feature_type = str(raw_type).strip().lower()
    if feature_type == "torsion":
        feature_type = "dihedral"

    atoms_obj = (
        entry_map.get("atom_indices")
        or entry_map.get("atoms")
        or entry_map.get("indices")
    )
    if atoms_obj is None:
        raise FeatureSpecError(
            "feature entry missing 'atom_indices' / 'atoms' / 'indices'"
        )
    if not isinstance(atoms_obj, Sequence):
        raise FeatureSpecError("feature atoms must be a sequence of integers")
    atoms = [int(v) for v in atoms_obj]

    name_obj = entry_map.get("name")
    feature_name = str(name_obj) if name_obj is not None else f"feature_{position}"
    weight = _coerce_weight(entry_map.get("weight", 1.0), feature_name)
    pbc_flag = bool(entry_map.get("pbc", default_pbc))
    return _FeatureRecord(
        name=feature_name,
        feature_type=feature_type,
        atoms=atoms,
        weight=weight,
        pbc=pbc_flag,
        position=position,
    )


def _normalise_spec_input(
    spec: Mapping[str, object] | Sequence[Mapping[str, object]],
) -> tuple[Sequence[Mapping[str, object]], bool]:
    if isinstance(spec, Mapping):
        feature_entries = spec.get("features")
        if feature_entries is None:
            raise FeatureSpecError("feature spec mapping is missing 'features'")
        if not isinstance(feature_entries, Sequence):
            raise FeatureSpecError("feature list must be a sequence")
        return feature_entries, bool(spec.get("use_pbc", True))
    if not isinstance(spec, Sequence):
        raise FeatureSpecError("feature list must be a sequence")
    return spec, True


class _FeatureCollector:
    def __init__(self, default_pbc: bool) -> None:
        self._default_pbc = default_pbc
        self.feature_names: list[str] = []
        self.feature_weights: list[float] = []
        self.distance_pairs: list[list[int]] = []
        self.distance_positions: list[int] = []
        self.distance_pbc_flags: list[bool] = []
        self.angle_triplets: list[list[int]] = []
        self.angle_positions: list[int] = []
        self.angle_pbc_flags: list[bool] = []
        self.dihedral_quads: list[list[int]] = []
        self.dihedral_positions: list[int] = []
        self.dihedral_pbc_flags: list[bool] = []
        self._max_index = -1
        self._any_pbc = False
        self._handlers = {
            "distance": self._add_distance,
            "angle": self._add_angle,
            "dihedral": self._add_dihedral,
        }

    def add_entry(self, entry: Mapping[str, object], position: int) -> None:
        record = _parse_feature_entry(entry, position, self._default_pbc)
        self._register_metadata(record)
        handler = self._handlers.get(record.feature_type)
        if handler is None:
            raise FeatureSpecError(f"unsupported feature type '{record.feature_type}'")
        handler(record)

    def _register_metadata(self, record: _FeatureRecord) -> None:
        self.feature_names.append(record.name)
        self.feature_weights.append(record.weight)
        self._any_pbc = self._any_pbc or record.pbc
        if record.atoms:
            self._max_index = max(self._max_index, max(record.atoms))

    def _add_distance(self, record: _FeatureRecord) -> None:
        atoms = _as_int_sequence(record.atoms, 2, label="distance")
        self.distance_pairs.append(atoms)
        self.distance_positions.append(record.position)
        self.distance_pbc_flags.append(record.pbc)

    def _add_angle(self, record: _FeatureRecord) -> None:
        atoms = _as_int_sequence(record.atoms, 3, label="angle")
        self.angle_triplets.append(atoms)
        self.angle_positions.append(record.position)
        self.angle_pbc_flags.append(record.pbc)

    def _add_dihedral(self, record: _FeatureRecord) -> None:
        atoms = _as_int_sequence(record.atoms, 4, label="dihedral")
        self.dihedral_quads.append(atoms)
        self.dihedral_positions.append(record.position)
        self.dihedral_pbc_flags.append(record.pbc)

    @staticmethod
    def _tensor_or_empty(
        values: Sequence[Sequence[int]] | Sequence[int] | Sequence[bool],
        *,
        dtype: torch.dtype,
        device: torch.device,
        width: int | None = None,
    ) -> Tensor:
        if values:
            return torch.as_tensor(values, dtype=dtype, device=device)
        if width is None:
            return torch.zeros((0,), dtype=dtype, device=device)
        return torch.zeros((0, width), dtype=dtype, device=device)

    def build(self) -> NormalizedFeatureSpec:
        if not self.feature_names:
            raise FeatureSpecError(
                "feature specification must contain at least one entry"
            )

        atom_count = self._max_index + 1
        if atom_count <= 0:
            raise FeatureSpecError("feature specification references no atoms")

        device = torch.device("cpu")
        long_dtype = torch.long

        distance_pairs_tensor = self._tensor_or_empty(
            self.distance_pairs, dtype=long_dtype, device=device, width=2
        )
        angle_triplets_tensor = self._tensor_or_empty(
            self.angle_triplets, dtype=long_dtype, device=device, width=3
        )
        dihedral_quads_tensor = self._tensor_or_empty(
            self.dihedral_quads, dtype=long_dtype, device=device, width=4
        )

        distance_pos_tensor = self._tensor_or_empty(
            self.distance_positions, dtype=long_dtype, device=device
        )
        angle_pos_tensor = self._tensor_or_empty(
            self.angle_positions, dtype=long_dtype, device=device
        )
        dihedral_pos_tensor = self._tensor_or_empty(
            self.dihedral_positions, dtype=long_dtype, device=device
        )

        distance_pbc_tensor = self._tensor_or_empty(
            self.distance_pbc_flags, dtype=torch.bool, device=device
        )
        angle_pbc_tensor = self._tensor_or_empty(
            self.angle_pbc_flags, dtype=torch.bool, device=device
        )
        dihedral_pbc_tensor = self._tensor_or_empty(
            self.dihedral_pbc_flags, dtype=torch.bool, device=device
        )

        return NormalizedFeatureSpec(
            feature_names=tuple(self.feature_names),
            use_pbc=bool(self._any_pbc),
            atom_count=int(atom_count),
            distance_pairs=distance_pairs_tensor,
            distance_positions=distance_pos_tensor,
            distance_pbc=distance_pbc_tensor,
            angle_triplets=angle_triplets_tensor,
            angle_positions=angle_pos_tensor,
            angle_pbc=angle_pbc_tensor,
            dihedral_quads=dihedral_quads_tensor,
            dihedral_positions=dihedral_pos_tensor,
            dihedral_pbc=dihedral_pbc_tensor,
            feature_weights=torch.as_tensor(
                self.feature_weights, dtype=torch.float32, device=device
            ),
        )


def canonicalize_feature_spec(
    spec: Mapping[str, object] | Sequence[Mapping[str, object]],
) -> NormalizedFeatureSpec:
    """
    Canonicalise a user provided feature specification.

    Parameters
    ----------
    spec
        Either a mapping with ``\"features\"`` and optional metadata or a plain
        sequence of feature mappings.  Each feature mapping must include:
            - ``type``: one of ``distance``, ``angle`` or ``dihedral``
            - ``atoms`` or ``indices``: iterable of atom indices
            - optional ``name``: string identifier, auto-generated if omitted

    Returns
    -------
    NormalizedFeatureSpec
        Structured representation ready to be embedded inside TorchScript.
    """

    features_iter, default_pbc = _normalise_spec_input(spec)
    collector = _FeatureCollector(default_pbc)
    for position, entry in enumerate(features_iter):
        collector.add_entry(entry, position)
    return collector.build()


class TorchscriptFeatureExtractor(nn.Module):
    """Vectorised TorchScript implementation of the feature extraction pipeline."""

    def __init__(self, spec: NormalizedFeatureSpec) -> None:
        super().__init__()
        self.feature_count: int = int(spec.n_features)
        self.atom_count: int = int(spec.atom_count)
        self.use_pbc: bool = bool(spec.use_pbc)

        self.register_buffer("distance_pairs", spec.distance_pairs)
        self.register_buffer("distance_positions", spec.distance_positions)
        self.register_buffer("distance_pbc_mask", spec.distance_pbc)
        self.register_buffer("angle_triplets", spec.angle_triplets)
        self.register_buffer("angle_positions", spec.angle_positions)
        self.register_buffer("angle_pbc_mask", spec.angle_pbc)
        self.register_buffer("dihedral_quads", spec.dihedral_quads)
        self.register_buffer("dihedral_positions", spec.dihedral_positions)
        self.register_buffer("dihedral_pbc_mask", spec.dihedral_pbc)
        self.register_buffer("feature_weights", spec.feature_weights)
        self.register_buffer("eps", torch.tensor(1.0e-12, dtype=torch.float32))

    def forward(self, positions: Tensor, box: Tensor) -> Tensor:  # noqa: D401
        """
        Transform atomic positions into molecular features.

        Parameters
        ----------
        positions
            Atomic positions in nanometres, shape ``(N, 3)``.
        box
            Periodic box vectors in nanometres, shape ``(3, 3)``.

        Returns
        -------
        torch.Tensor
            Feature vector of length ``F`` in the canonical order defined by the
            input specification.
        """
        if positions.dim() != 2 or positions.size(-1) != 3:
            raise RuntimeError("positions must have shape (N, 3)")
        if positions.size(0) < self.atom_count:
            raise RuntimeError(
                f"positions tensor has {positions.size(0)} atoms, "
                f"but specification references {self.atom_count}"
            )
        if box.dim() != 2 or box.size(0) != 3 or box.size(1) != 3:
            raise RuntimeError("box tensor must have shape (3, 3)")

        pos = positions.to(dtype=torch.float32)
        box32 = box.to(dtype=torch.float32)

        features = torch.zeros(
            self.feature_count, dtype=torch.float32, device=pos.device
        )

        if self.use_pbc:
            inv_box = torch.inverse(box32)
        else:
            inv_box = torch.eye(3, dtype=torch.float32, device=pos.device)

        if self.distance_pairs.size(0) > 0:
            values = self._compute_distances(pos, box32, inv_box)
            features = features.index_put_((self.distance_positions,), values)

        if self.angle_triplets.size(0) > 0:
            values = self._compute_angles(pos, box32, inv_box)
            features = features.index_put_((self.angle_positions,), values)

        if self.dihedral_quads.size(0) > 0:
            values = self._compute_dihedrals(pos, box32, inv_box)
            features = features.index_put_((self.dihedral_positions,), values)

        return features * self.feature_weights

    def _apply_minimum_image(
        self, vectors: Tensor, box: Tensor, inv_box: Tensor
    ) -> Tensor:
        if not self.use_pbc:
            return vectors
        fractional = torch.matmul(vectors, inv_box)
        wrapped = fractional - torch.round(fractional)
        return torch.matmul(wrapped, box)

    def _compute_distances(self, pos: Tensor, box: Tensor, inv_box: Tensor) -> Tensor:
        idx_i = self.distance_pairs[:, 0]
        idx_j = self.distance_pairs[:, 1]
        vec = pos.index_select(0, idx_j) - pos.index_select(0, idx_i)
        wrapped = self._apply_minimum_image(vec, box, inv_box)
        if self.distance_pbc_mask.numel() > 0:
            disp = torch.where(self.distance_pbc_mask.unsqueeze(-1), wrapped, vec)
        else:
            disp = wrapped
        squared = torch.sum(disp * disp, dim=-1)
        return torch.sqrt(torch.clamp_min(squared, self.eps)).to(torch.float32)

    def _compute_angles(self, pos: Tensor, box: Tensor, inv_box: Tensor) -> Tensor:
        idx_i = self.angle_triplets[:, 0]
        idx_j = self.angle_triplets[:, 1]
        idx_k = self.angle_triplets[:, 2]

        vec_ij = pos.index_select(0, idx_i) - pos.index_select(0, idx_j)
        vec_kj = pos.index_select(0, idx_k) - pos.index_select(0, idx_j)

        v1_wrap = self._apply_minimum_image(vec_ij, box, inv_box)
        v2_wrap = self._apply_minimum_image(vec_kj, box, inv_box)
        if self.angle_pbc_mask.numel() > 0:
            mask = self.angle_pbc_mask.unsqueeze(-1)
            v1 = torch.where(mask, v1_wrap, vec_ij)
            v2 = torch.where(mask, v2_wrap, vec_kj)
        else:
            v1 = v1_wrap
            v2 = v2_wrap

        dot = torch.sum(v1 * v2, dim=-1)
        n1 = torch.sqrt(torch.clamp_min(torch.sum(v1 * v1, dim=-1), self.eps))
        n2 = torch.sqrt(torch.clamp_min(torch.sum(v2 * v2, dim=-1), self.eps))
        cos_theta = dot / (n1 * n2)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        return torch.acos(cos_theta)

    def _compute_dihedrals(self, pos: Tensor, box: Tensor, inv_box: Tensor) -> Tensor:
        idx_i = self.dihedral_quads[:, 0]
        idx_j = self.dihedral_quads[:, 1]
        idx_k = self.dihedral_quads[:, 2]
        idx_l = self.dihedral_quads[:, 3]

        b0_raw = pos.index_select(0, idx_j) - pos.index_select(0, idx_i)
        b1_raw = pos.index_select(0, idx_k) - pos.index_select(0, idx_j)
        b2_raw = pos.index_select(0, idx_l) - pos.index_select(0, idx_k)

        b0_wrap = self._apply_minimum_image(b0_raw, box, inv_box)
        b1_wrap = self._apply_minimum_image(b1_raw, box, inv_box)
        b2_wrap = self._apply_minimum_image(b2_raw, box, inv_box)

        if self.dihedral_pbc_mask.numel() > 0:
            mask = self.dihedral_pbc_mask.unsqueeze(-1)
            b0 = torch.where(mask, b0_wrap, b0_raw)
            b1 = torch.where(mask, b1_wrap, b1_raw)
            b2 = torch.where(mask, b2_wrap, b2_raw)
        else:
            b0 = b0_wrap
            b1 = b1_wrap
            b2 = b2_wrap

        c0 = torch.cross(b0, b1, dim=-1)
        c1 = torch.cross(b1, b2, dim=-1)

        c0_norm = torch.sqrt(torch.clamp_min(torch.sum(c0 * c0, dim=-1), self.eps))
        c1_norm = torch.sqrt(torch.clamp_min(torch.sum(c1 * c1, dim=-1), self.eps))
        b1_norm = torch.sqrt(torch.clamp_min(torch.sum(b1 * b1, dim=-1), self.eps))

        c0 = c0 / c0_norm.unsqueeze(-1)
        c1 = c1 / c1_norm.unsqueeze(-1)
        b1_unit = b1 / b1_norm.unsqueeze(-1)

        x = torch.sum(c0 * c1, dim=-1)
        y = torch.sum(torch.cross(c0, c1, dim=-1) * b1_unit, dim=-1)
        mask = (torch.abs(x) + torch.abs(y)) >= self.eps
        safe_x = torch.where(mask, x, torch.ones_like(x))
        safe_y = torch.where(mask, y, torch.zeros_like(y))
        angle = torch.atan2(safe_y, safe_x)
        return torch.where(mask, angle, torch.zeros_like(angle))


def build_feature_extractor_module(
    spec: NormalizedFeatureSpec,
) -> TorchscriptFeatureExtractor:
    """Instantiate the TorchScript-ready feature extractor for the provided spec."""

    return TorchscriptFeatureExtractor(spec)
