from __future__ import annotations

from typing import List, Optional, Protocol

import numpy as np


class _SupportsTRAM(Protocol):
    temperatures: List[float]
    dtrajs: List[np.ndarray]
    transition_matrix: Optional[np.ndarray]
    count_matrix: Optional[np.ndarray]
    stationary_distribution: Optional[np.ndarray]

    def _build_standard_msm(
        self, lag_time: int, count_mode: str = "sliding"
    ) -> None: ...


class TRAMMixin:
    def _build_tram_msm(self: _SupportsTRAM, lag_time: int) -> None:
        import logging as _logging

        _logging.getLogger("pmarlo").info(
            "Building TRAM MSM for multi-temperature data via deeptime..."
        )
        if len(self.temperatures) <= 1:
            raise ValueError("TRAM MSM requires at least two ensembles")

        import numpy as _np
        from deeptime.markov.msm import TRAM, TRAMDataset  # type: ignore

        bias = getattr(self, "bias_matrices", None)
        if bias is None:
            raise ValueError("Bias matrices are required for TRAM MSM construction")

        ds = TRAMDataset(dtrajs=self.dtrajs, bias_matrices=bias)  # type: ignore[call-arg]
        tram = TRAM(
            lagtime=int(max(1, lag_time)),
            count_mode="sliding",
            init_strategy="MBAR",
        )
        tram_model = tram.fit(ds).fetch_model()
        ref = int(getattr(self, "tram_reference_index", 0))
        msms = getattr(tram_model, "msms", None)
        cm_list = getattr(tram_model, "count_models", None)

        if isinstance(msms, list) and 0 <= ref < len(msms):
            msm_ref = msms[ref]
            self.transition_matrix = _np.asarray(msm_ref.transition_matrix, dtype=float)
            stationary = getattr(msm_ref, "stationary_distribution", None)
            if stationary is not None:
                self.stationary_distribution = _np.asarray(stationary, dtype=float)

            if isinstance(cm_list, list) and 0 <= ref < len(cm_list):
                self.count_matrix = _np.asarray(
                    cm_list[ref].count_matrix, dtype=float
                )
            return

        msm_collection = getattr(tram_model, "msm_collection", None)
        if msm_collection is None:
            raise RuntimeError("TRAM did not expose per-ensemble MSMs")

        if hasattr(msm_collection, "select"):
            msm_collection.select(ref)

        transition = getattr(msm_collection, "transition_matrix", None)
        if transition is None:
            raise RuntimeError("TRAM did not yield a transition matrix")
        self.transition_matrix = _np.asarray(transition, dtype=float)

        stationary = getattr(msm_collection, "stationary_distribution", None)
        if stationary is not None:
            self.stationary_distribution = _np.asarray(stationary, dtype=float)

        count_model = getattr(msm_collection, "count_model", None)
        if count_model is not None and hasattr(count_model, "count_matrix"):
            self.count_matrix = _np.asarray(count_model.count_matrix, dtype=float)
