"""Public API facade for PMARLO.

This module re-exports the high-level helpers that historically lived inside the
monolithic ``pmarlo.api`` module.  Existing callers continue to import from
``pmarlo.api`` while the implementation now lives across dedicated submodules.
"""

from .clustering import cluster_microstates
from .conformations import (
    find_conformations,
    find_conformations_with_msm,
    sanitize_label_for_filename,
)
from .demux import demultiplex_run
from .features import (
    align_trajectory,
    compute_features,
    compute_universal_embedding,
    compute_universal_metric,
    normalize_training_metrics,
    reduce_features,
    trig_expand_periodic,
    _fes_build_phi_psi_maps,
    _fes_pair_from_phi_psi_maps,
)
from .fes import (
    generate_free_energy_surface,
    generate_fes_and_pick_minima,
    select_fes_pair,
)
from .msm import (
    analyze_msm,
    build_msm_from_labels,
    compute_macrostates,
    macro_mfpt,
    macro_transition_matrix,
    macrostate_populations,
)
from .replica_exchange import run_replica_exchange
from .shards import (
    build_from_shards,
    emit_shards_rg_rmsd,
    emit_shards_rg_rmsd_windowed,
    select_shard_paths,
    _build_opts,
)
from .trajectory_utils import (
    extract_last_frame_to_pdb,
    extract_random_highT_frame_to_pdb,
)
from .workflow import build_joint_workflow
from pmarlo.reweight import normalize_reweight_mode
from pmarlo.utils.config_utils import deep_merge, resolve_deeptica
from pmarlo.utils.input_parsing import parse_temperature_ladder, parse_tau_schedule, coerce_tau_schedule, parse_bins, parse_hidden_layers
from pmarlo.utils.json_io import sanitize, write_json, sanitize_deeptica_payload
from pmarlo.utils.naming import slugify, timestamp
from pmarlo.utils.path_utils import coerce_path_list, relativize
from pmarlo.utils.seed import choose_sim_seed, extract_seed

# Backward compatibility for code importing the underscored helper directly.
_trig_expand_periodic = trig_expand_periodic

__all__ = [
    "align_trajectory",
    "analyze_msm",
    "build_from_shards",
    "build_joint_workflow",
    "choose_sim_seed",
    "build_msm_from_labels",
    "cluster_microstates",
    "coerce_path_list",
    "coerce_tau_schedule",
    "compute_features",
    "compute_macrostates",
    "compute_universal_embedding",
    "compute_universal_metric",
    "deep_merge",
    "demultiplex_run",
    "emit_shards_rg_rmsd",
    "emit_shards_rg_rmsd_windowed",
    "extract_last_frame_to_pdb",
    "extract_random_highT_frame_to_pdb",
    "extract_seed",
    "find_conformations",
    "find_conformations_with_msm",
    "generate_free_energy_surface",
    "generate_fes_and_pick_minima",
    "macro_mfpt",
    "macro_transition_matrix",
    "macrostate_populations",
    "normalize_training_metrics",
    "normalize_reweight_mode",
    "parse_bins",
    "parse_hidden_layers",
    "parse_tau_schedule",
    "parse_temperature_ladder",
    "reduce_features",
    "relativize",
    "resolve_deeptica",
    "run_replica_exchange",
    "sanitize",
    "sanitize_deeptica_payload",
    "sanitize_label_for_filename",
    "select_fes_pair",
    "select_shard_paths",
    "slugify",
    "timestamp",
    "trig_expand_periodic",
    "write_json",
    "_trig_expand_periodic",
    "_build_opts",
    "_fes_build_phi_psi_maps",
    "_fes_pair_from_phi_psi_maps",
]
