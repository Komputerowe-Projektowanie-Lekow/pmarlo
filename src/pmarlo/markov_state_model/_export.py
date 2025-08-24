from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


class ExportMixin:
    def save_analysis_results(self, prefix: str = "msm_analysis") -> None:
        import json as _json
        import logging as _logging
        import pickle as _pickle

        _logging.getLogger("pmarlo").info("Saving analysis results...")

        # Core matrices
        self._save_transition_matrix(prefix)
        self._save_count_matrix(prefix)

        # Scalars and distributions
        self._save_free_energies(prefix)
        self._save_stationary_distribution(prefix)

        # Trajectories and tables
        self._save_discrete_trajectories(prefix)
        self._save_state_table_file(prefix)
        self._save_free_energy_bar_plot(prefix)

        # FES
        self._save_fes_array(prefix)

        # Structured results
        analysis_results: Dict[str, Any] = {}
        from pmarlo.results import FESResult, MSMResult

        if (
            getattr(self, "transition_matrix", None) is not None
            and getattr(self, "count_matrix", None) is not None
        ):
            analysis_results["msm"] = MSMResult(
                transition_matrix=self.transition_matrix,
                count_matrix=self.count_matrix,
                free_energies=getattr(self, "free_energies", None),
                stationary_distribution=getattr(self, "stationary_distribution", None),
            )
        if getattr(self, "fes_data", None) is not None:
            analysis_results["fes"] = FESResult(
                free_energy=self.fes_data["free_energy"],
                xedges=self.fes_data["xedges"],
                yedges=self.fes_data["yedges"],
                cv1_name=self.fes_data["cv1_name"],
                cv2_name=self.fes_data["cv2_name"],
                temperature=self.fes_data["temperature"],
            )
        if getattr(self, "implied_timescales", None) is not None:
            analysis_results["its"] = self.implied_timescales

        results_file = self.output_dir / "analysis_results.pkl"
        json_file = self.output_dir / "analysis_results.json"
        with results_file.open("wb") as f:
            _pickle.dump(analysis_results, f)
        with json_file.open("w") as f:
            _json.dump(
                {k: v.to_dict(metadata_only=True) for k, v in analysis_results.items()},
                f,
            )

        self._log_save_completion()

    def _save_transition_matrix(self, prefix: str) -> None:
        self._save_matrix_intelligent(
            self.transition_matrix, "transition_matrix", prefix
        )

    def _save_count_matrix(self, prefix: str) -> None:
        self._save_matrix_intelligent(self.count_matrix, "count_matrix", prefix)

    def _save_free_energies(self, prefix: str) -> None:
        if getattr(self, "free_energies", None) is not None:
            np.save(self.output_dir / f"{prefix}_free_energies.npy", self.free_energies)

    def _save_stationary_distribution(self, prefix: str) -> None:
        if getattr(self, "stationary_distribution", None) is not None:
            np.save(
                self.output_dir / f"{prefix}_stationary_distribution.npy",
                self.stationary_distribution,
            )

    def _save_discrete_trajectories(self, prefix: str) -> None:
        if not getattr(self, "dtrajs", []):
            return
        try:
            dtrajs_obj = np.array(self.dtrajs, dtype=object)
            np.save(self.output_dir / f"{prefix}_dtrajs.npy", dtrajs_obj)
        except Exception:
            for idx, dtraj in enumerate(self.dtrajs):
                try:
                    np.save(
                        self.output_dir / f"{prefix}_dtrajs_traj{idx:02d}.npy",
                        np.asarray(dtraj),
                    )
                except Exception:
                    continue

    def _save_state_table_file(self, prefix: str) -> None:
        if getattr(self, "state_table", None) is not None:
            self.state_table.to_csv(
                self.output_dir / f"{prefix}_state_table.csv", index=False
            )

    def _save_fes_array(self, prefix: str) -> None:
        if getattr(self, "fes_data", None) is None:
            return
        np.save(self.output_dir / f"{prefix}_fes.npy", self.fes_data["free_energy"])

    def _save_free_energy_bar_plot(self, prefix: str) -> None:
        if getattr(self, "state_table", None) is None:
            return
        try:
            import matplotlib.pyplot as _plt
        except Exception:
            return
        fe = self.state_table.get("free_energy_kJ_mol")
        if fe is None:
            return
        err = self.state_table.get("free_energy_error", pd.Series(np.zeros(len(fe))))
        fig, ax = _plt.subplots()
        ax.bar(self.state_table["state_id"], fe, yerr=err, capsize=4)
        ax.set_xlabel("State")
        ax.set_ylabel("Free energy (kJ/mol)")
        fig.tight_layout()
        fig.savefig(self.output_dir / f"{prefix}_free_energy_bar.png")
        _plt.close(fig)

    def _log_save_completion(self) -> None:
        import logging as _logging

        _logging.getLogger("pmarlo").info(
            f"Analysis results saved to {self.output_dir}"
        )
