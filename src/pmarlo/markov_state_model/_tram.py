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
            _logging.getLogger("pmarlo").warning(
                "Only one ensemble provided, falling back to standard MSM"
            )
            # fallback provided by EstimationMixin
            self._build_standard_msm(lag_time)
            return

        try:
            from deeptime.markov.msm import TRAM, TRAMDataset  # type: ignore

            bias = getattr(self, "bias_matrices", None)
            if bias is None:
                _logging.getLogger("pmarlo").warning(
                    "No bias matrices provided for TRAM; falling back to standard MSM."
                )
                self._build_standard_msm(lag_time)
                return
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
            import numpy as _np

            if isinstance(msms, list) and 0 <= ref < len(msms):
                msm_ref = msms[ref]
                self.transition_matrix = _np.asarray(
                    msm_ref.transition_matrix, dtype=float
                )
                if (
                    hasattr(msm_ref, "stationary_distribution")
                    and msm_ref.stationary_distribution is not None
                ):
                    self.stationary_distribution = _np.asarray(
                        msm_ref.stationary_distribution, dtype=float
                    )
                if isinstance(cm_list, list) and 0 <= ref < len(cm_list):
                    self.count_matrix = _np.asarray(
                        cm_list[ref].count_matrix, dtype=float
                    )
            else:
                _logging.getLogger("pmarlo").warning(
                    "TRAM did not expose per-ensemble MSMs; falling back to standard MSM"
                )
                self._build_standard_msm(lag_time)
                return
        except Exception as e:
            _logging.getLogger("pmarlo").warning(
                f"deeptime TRAM unavailable or failed ({e}); using standard MSM"
            )
            self._build_standard_msm(lag_time)
            return
