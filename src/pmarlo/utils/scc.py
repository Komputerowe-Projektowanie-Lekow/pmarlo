"""Utilities for analysing strongly connected components in directed graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import strongly_connected_components

__all__ = ["SCCSummary", "analyse_scc", "compute_component_coverage"]


@dataclass(frozen=True, slots=True)
class SCCSummary:
    """Summary information about the SCC structure of a directed graph."""

    n_nodes: int
    component_labels: np.ndarray
    components: list[np.ndarray]
    component_sizes: np.ndarray
    largest_component: np.ndarray
    largest_fraction: float | None
    state_indices: np.ndarray

    def components_global(self) -> list[np.ndarray]:
        """Return component members mapped to the provided state indices."""

        base = self.state_indices
        return [base[idx] for idx in self.components]

    def largest_component_global(self) -> np.ndarray:
        """Return indices of the largest component mapped to the state indices."""

        return self.state_indices[self.largest_component]

    def component_sizes_sorted(self) -> np.ndarray:
        """Return component sizes sorted in descending order."""

        if self.component_sizes.size == 0:
            return self.component_sizes
        return np.sort(self.component_sizes)[::-1]

    def to_artifact(self, *, coverage: float | None = None) -> dict[str, Any]:
        """Return a JSON-friendly payload describing the SCC structure."""

        payload: dict[str, Any] = {
            "n_states": int(self.n_nodes),
            "component_sizes": self.component_sizes.astype(int).tolist(),
            "component_sizes_sorted": self.component_sizes_sorted()
            .astype(int)
            .tolist(),
            "largest_component_size": int(self.largest_component.size),
            "largest_component_fraction_states": (
                float(self.largest_fraction)
                if self.largest_fraction is not None
                else None
            ),
            "largest_component_indices": self.largest_component_global()
            .astype(int)
            .tolist(),
        }
        if coverage is not None:
            payload["largest_component_fraction_frames"] = float(coverage)
        return payload


def analyse_scc(
    counts: np.ndarray,
    *,
    state_indices: Iterable[int] | None = None,
) -> SCCSummary:
    """Analyse the SCC structure of a transition count matrix."""

    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError("count matrix must be square")

    n = int(counts.shape[0])
    if state_indices is None:
        state_indices = np.arange(n, dtype=int)
    else:
        state_indices = np.asarray(list(state_indices), dtype=int).reshape(n)

    if n == 0:
        empty = np.empty((0,), dtype=int)
        return SCCSummary(
            n_nodes=0,
            component_labels=empty,
            components=[],
            component_sizes=empty,
            largest_component=empty,
            largest_fraction=None,
            state_indices=empty,
        )

    adjacency_matrix = csr_matrix(counts > 0.0)
    n_components, labels = strongly_connected_components(
        csgraph=adjacency_matrix, directed=True, connection="strong"
    )

    components: list[np.ndarray] = []
    component_sizes: list[int] = []
    for comp_idx in range(n_components):
        members = np.where(labels == comp_idx)[0]
        if members.size == 0:
            continue
        arr = np.array(sorted(members), dtype=int)
        components.append(arr)
        component_sizes.append(int(arr.size))

    if component_sizes:
        sizes_arr = np.asarray(component_sizes, dtype=int)
        idx_largest = int(np.argmax(sizes_arr))
        largest = components[idx_largest]
        fraction = float(sizes_arr[idx_largest] / n)
    else:
        sizes_arr = np.empty((0,), dtype=int)
        largest = np.empty((0,), dtype=int)
        fraction = None

    return SCCSummary(
        n_nodes=n,
        component_labels=np.asarray(labels, dtype=int),
        components=components,
        component_sizes=sizes_arr,
        largest_component=largest,
        largest_fraction=fraction,
        state_indices=np.asarray(state_indices, dtype=int),
    )


def compute_component_coverage(
    population: np.ndarray,
    component_indices: Sequence[int],
) -> float | None:
    """Compute coverage of a component given per-state populations."""

    if population.size == 0:
        return None
    total = float(np.sum(population))
    if total <= 0.0:
        return None
    component_total = float(
        np.sum(population[np.asarray(component_indices, dtype=int)])
    )
    return component_total / total
