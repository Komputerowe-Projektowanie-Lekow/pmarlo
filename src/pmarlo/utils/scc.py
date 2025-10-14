"""Utilities for analysing strongly connected components in directed graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

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


class _TarjanSCC:
    """Tarjan strongly connected components solver."""

    def __init__(self, adjacency: Sequence[Sequence[int]]) -> None:
        self.adjacency = adjacency
        self.n = len(adjacency)
        self.index = 0
        self.indices = np.full(self.n, -1, dtype=int)
        self.lowlinks = np.zeros(self.n, dtype=int)
        self.on_stack = np.zeros(self.n, dtype=bool)
        self.stack: list[int] = []
        self.components: list[list[int]] = []
        self.labels = np.full(self.n, -1, dtype=int)

    def run(self) -> tuple[list[list[int]], np.ndarray]:
        for v in range(self.n):
            if self.indices[v] == -1:
                self._visit(v)
        return self.components, self.labels

    def _visit(self, v: int) -> None:
        self.indices[v] = self.index
        self.lowlinks[v] = self.index
        self.index += 1
        self.stack.append(v)
        self.on_stack[v] = True

        for w in self.adjacency[v]:
            if self.indices[w] == -1:
                self._visit(w)
                self.lowlinks[v] = min(self.lowlinks[v], self.lowlinks[w])
            elif self.on_stack[w]:
                self.lowlinks[v] = min(self.lowlinks[v], self.indices[w])

        if self.lowlinks[v] == self.indices[v]:
            component: list[int] = []
            while True:
                w = self.stack.pop()
                self.on_stack[w] = False
                component.append(w)
                if w == v:
                    break
            comp_idx = len(self.components)
            for node in component:
                self.labels[node] = comp_idx
            self.components.append(component)


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
        empty = np.zeros((0,), dtype=int)
        return SCCSummary(
            n_nodes=0,
            component_labels=empty,
            components=[],
            component_sizes=empty,
            largest_component=empty,
            largest_fraction=None,
            state_indices=empty,
        )

    adjacency: list[list[int]] = [
        np.where(counts[i] > 0.0)[0].astype(int).tolist() for i in range(n)
    ]
    solver = _TarjanSCC(adjacency)
    components_raw, labels = solver.run()

    components: list[np.ndarray] = []
    component_sizes: list[int] = []
    for comp in components_raw:
        if not comp:
            continue
        arr = np.array(sorted(comp), dtype=int)
        components.append(arr)
        component_sizes.append(int(arr.size))

    if component_sizes:
        sizes_arr = np.asarray(component_sizes, dtype=int)
        idx_largest = int(np.argmax(sizes_arr))
        largest = components[idx_largest]
        fraction = float(sizes_arr[idx_largest] / n)
    else:
        sizes_arr = np.zeros((0,), dtype=int)
        largest = np.zeros((0,), dtype=int)
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
