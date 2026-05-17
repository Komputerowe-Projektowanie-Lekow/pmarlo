"""Public high-level API helpers for PMARLO."""

from pmarlo.features.deeptica.config import resolve_deeptica, sanitize_deeptica_payload
from pmarlo.utils.config_utils import deep_merge
from pmarlo.utils.input_parsing import (
    parse_bins,
    parse_hidden_layers,
    parse_tau_schedule,
    parse_temperature_ladder,
)
from pmarlo.utils.json_io import sanitize, write_json
from pmarlo.utils.naming import slugify, timestamp
from pmarlo.utils.path_utils import coerce_path_list, relativize
from pmarlo.utils.seed import choose_sim_seed, extract_seed

from .clustering import cluster_microstates
from .conformations import (
    find_conformations,
    find_conformations_from_msm,
    sanitize_label_for_filename,
)
from .feature_profiles import (
    FEATURE_PROFILES,
    FeatureProfile,
    get_feature_profile_info,
    load_feature_profile,
    validate_profile_for_cv_biasing,
)
from .features import (
    align_trajectory,
    compute_features,
    compute_universal_embedding,
    compute_universal_metric,
    normalize_training_metrics,
    reduce_features,
    trig_expand_periodic,
)
from .fes import (
    generate_fes_and_pick_minima,
    generate_free_energy_surface,
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
from .trajectory_utils import extract_last_frame_to_pdb

__all__ = [
    "align_trajectory",
    "analyze_msm",
    "choose_sim_seed",
    "build_msm_from_labels",
    "cluster_microstates",
    "coerce_path_list",
    "compute_features",
    "compute_macrostates",
    "compute_universal_embedding",
    "compute_universal_metric",
    "deep_merge",
    "extract_last_frame_to_pdb",
    "extract_seed",
    "FEATURE_PROFILES",
    "FeatureProfile",
    "find_conformations",
    "find_conformations_from_msm",
    "generate_free_energy_surface",
    "generate_fes_and_pick_minima",
    "get_feature_profile_info",
    "load_feature_profile",
    "macro_mfpt",
    "macro_transition_matrix",
    "macrostate_populations",
    "normalize_training_metrics",
    "parse_bins",
    "parse_hidden_layers",
    "parse_tau_schedule",
    "parse_temperature_ladder",
    "reduce_features",
    "relativize",
    "resolve_deeptica",
    "sanitize",
    "sanitize_deeptica_payload",
    "sanitize_label_for_filename",
    "select_fes_pair",
    "slugify",
    "timestamp",
    "trig_expand_periodic",
    "validate_profile_for_cv_biasing",
    "write_json",
]
