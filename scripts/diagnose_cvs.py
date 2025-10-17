from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import mdtraj as md  # type: ignore # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pmarlo.io import trajectory as traj_io
from pmarlo.utils.path_utils import ensure_directory


def compute_cvs(
    pdb_file: Path,
    trajectory_paths: Sequence[Path],
    *,
    reference: Path | None = None,
    stride: int = 1,
) -> pd.DataFrame:
    pdb_file = pdb_file.resolve()
    if not pdb_file.exists():
        raise FileNotFoundError(f"PDB topology not found: {pdb_file}")

    top = md.load(str(pdb_file))
    if reference is not None:
        reference_path = Path(reference).resolve()
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference structure not found: {reference_path}")
        ref = md.load(str(reference_path), top=str(pdb_file))[0]
    else:
        ref = top[0]

    ca_sel = top.topology.select("name CA")
    ca_sel = ca_sel if ca_sel.size else None

    rg_parts: list[np.ndarray] = []
    rmsd_parts: list[np.ndarray] = []

    for traj_path in trajectory_paths:
        traj_path = traj_path.resolve()
        if not traj_path.exists():
            raise FileNotFoundError(f"Trajectory not found: {traj_path}")
        for chunk in traj_io.iterload(
            str(traj_path),
            top=str(pdb_file),
            stride=int(max(1, stride)),
            chunk=1000,
        ):
            try:
                chunk = chunk.superpose(ref, atom_indices=ca_sel)
            except Exception:
                pass
            rg_chunk = md.compute_rg(chunk).astype(np.float64)
            rmsd_chunk = md.rmsd(chunk, ref, atom_indices=ca_sel).astype(np.float64)
            rg_parts.append(rg_chunk)
            rmsd_parts.append(rmsd_chunk)

    rg = np.concatenate(rg_parts) if rg_parts else np.empty((0,), dtype=np.float64)
    rmsd = (
        np.concatenate(rmsd_parts)
        if rmsd_parts
        else np.empty((0,), dtype=np.float64)
    )
    return pd.DataFrame({"Rg": rg, "RMSD_ref": rmsd})


def analyse_dataframe(df: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    ensure_directory(output_dir)
    pos_inf = {col: int(np.isposinf(df[col].to_numpy()).sum()) for col in df.columns}
    neg_inf = {col: int(np.isneginf(df[col].to_numpy()).sum()) for col in df.columns}
    stats = {
        "row_count": int(df.shape[0]),
        "nan_counts": df.isna().sum().astype(int).to_dict(),
        "posinf_counts": pos_inf,
        "neginf_counts": neg_inf,
    }

    describe = df.replace([np.inf, -np.inf], np.nan).describe(include="all")
    stats["describe"] = json.loads(describe.to_json())
    stats["inf_counts"] = {
        col: pos_inf.get(col, 0) + neg_inf.get(col, 0) for col in df.columns
    }

    finite_df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if not finite_df.empty:
        plt.figure(figsize=(8, 6))
        plt.scatter(finite_df["Rg"], finite_df["RMSD_ref"], s=5, alpha=0.6)
        plt.xlabel("Rg")
        plt.ylabel("RMSD_ref")
        plt.title("Cv Scatter Plot")
        scatter_path = output_dir / "cv_scatter_plot.png"
        plt.tight_layout()
        plt.savefig(scatter_path, dpi=200)
        plt.close()
        stats["scatter_plot"] = str(scatter_path.resolve())
    else:
        stats["scatter_plot"] = None

    return stats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose CV data (Rg, RMSD_ref).")
    parser.add_argument("--pdb", required=True, type=Path, help="Path to PDB topology.")
    parser.add_argument(
        "--traj",
        required=True,
        action="append",
        type=Path,
        help="Trajectory file to load (repeatable).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional reference structure for RMSD alignment.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Stride when loading.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cv_diagnostics"),
        help="Directory to store plots and summary JSON.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write JSON summary.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    df = compute_cvs(
        Path(args.pdb),
        [Path(p) for p in args.traj],
        reference=args.reference,
        stride=int(max(1, args.stride)),
    )
    stats = analyse_dataframe(df, Path(args.output_dir))

    print("=== CV Diagnostics ===", file=sys.stdout)
    print(f"Rows loaded: {stats['row_count']}", file=sys.stdout)
    print("NaN counts:", json.dumps(stats["nan_counts"], indent=2), file=sys.stdout)
    inf_counts = stats.get("inf_counts", {})
    print("Infinity counts:", json.dumps(inf_counts, indent=2), file=sys.stdout)
    describe_json = stats["describe"]
    print("Describe:", json.dumps(describe_json, indent=2), file=sys.stdout)
    if stats.get("scatter_plot"):
        print(f"Scatter plot saved to: {stats['scatter_plot']}", file=sys.stdout)
    else:
        print("Scatter plot skipped: no finite rows.", file=sys.stdout)

    if args.summary_json:
        summary_path = Path(args.summary_json)
        ensure_directory(summary_path.parent)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)
        print(f"Summary written to: {summary_path}", file=sys.stdout)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
