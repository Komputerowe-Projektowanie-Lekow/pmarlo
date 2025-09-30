Shards module:
- unnegotiable
- explanation
- key values
- best practices
- wanted outcomes
- examples
- files description
- maintenance and evolution


# Unnegotiable
1. Single canonical format only. Shards are emitted and consumed in one format; no heuristics, legacy fallbacks, or format guessing. There is no DEMUX and REPLICA shards. There only should be one shard which is DEMUX ones, which are specified especially for the ML/AI.
2. Deterministic emission. Same inputs equals same outputs (byte‑for‑byte for metadata, numerically stable arrays). No randomness, no time‑dependent behavior in content.
3. Kind purity. Pipelines and analyses operate on one shard kind at a time. Mixed kinds (e.g., ['demux', 'replica']) are invalid and must raise.
4. Strict schema validation. Every shard must validate against /shards/schema.py before being accepted by any downstream consumer.
5. Explicit temperature. temperature_K is mandatory where applicable and is treated as an exact filter in selection.
6. Stable identity and uniqueness. Duplicate DEMUX shards are disallowed. Enforce uniqueness using the composite key used by assemble.load_shards: (replica_id, segment_id, round(temperature_K, 3)).

# Small explanation
A shard is a uniform, self‑describing unit of simulation output prepared for dataset construction and ML/analysis workflows. Shards are typically emitted after Replica Exchange demultiplexing and are ready for direct consumption by downstream modules (e.g., MSM building or Deep‑TICA style learning) without ad‑hoc massaging.

# Key values
- Compatibility > convenience: downstream modules depend on exact structure; don’t improvise formats.
- Observability: all shards have auditable provenance in metadata.
- Reproducibility: emission and selection are pure, parameterized functions over the filesystem.
- Composability: selection and grouping operations are predictable and stable across runs.


# Best practices
1. Validate early: run schema validation at emit time and again at load time in CI and during data ingestion.
2. Freeze versions: bump a schema/version field when changing anything structural; never silent‑upgrade.
3. Guard uniqueness: when loading, maintain a seen‑set keyed by (replica_id, segment_id, round(temperature_K, 3)); raise on duplicates.
4. Filter explicitly: selection by temperature_K uses exact match; no closest‑temperature selection.
5. Unit discipline: all temperatures are Kelvin; document units in metadata fields. Avoid unit conversions inside emit.
6. Numerical precision: store arrays in a consistent dtype; do not mix float precisions within the same shard.
7. No cross‑contamination: a shard must reference one simulation segment; aggregation happens above this layer.

# Wanted outcomes
Uniform shards that pass schema checks and slot into MSM/Deep‑TICA without adapters.
Deterministic discovery/selection enabling exact reproduction of training/evaluation sets.
Clear failure modes (duplicate detection, mixed‑kind detection, schema failures) with actionable error messages.

# Examples
1. If the task involving some /ml/ module needs to be changed and that change also inlcude the /shards/ module, do not proceed with changing the shards module without any consultation, because all of the changes would included all of the created data before hand. This needs more recognition and not autonomous work on that.
2. ... TODO
3. ... TODO


# Files description

## assemble.py
Deterministically selects and loads shards based on metadata.
select_shards(root, temperature_K=None) - returns paths to shard JSONs under root. If temperature_K is provided, apply an exact filter using loaded metadata.
load_shards(json_paths) - loads NPZ+JSON pairs; validates schema; enforces no duplicates using the (replica_id, segment_id, round(temperature_K,3)) key; raises on violation.
group_by_temperature(shards) - groups loaded shards by meta.temperature_K and returns an ordered mapping.

Invariants:
- Returned shard sequence order is stable (sorted and reproducible).
- All returned shards share the same kind.

## discover.py
Filesystem discovery utilities.
discover_shard_jsons(root) - returns a sorted list of all shard JSON paths under root.
iter_metas(root) - yields parsed ShardMeta for each discovered JSON (validated but not fully loaded arrays).
list_temperatures(root) - returns unique sorted temperatures present in discovered metas.

Invariants:
- Discovery never mutates disk.
- Discovery never guesses formats; only recognizes the canonical one.

## emit.py
Shard writer. Emits exactly one canonical format.

Responsibilities:
- Assemble metadata that satisfies schema.py.
- Write arrays (e.g., coordinates, indices) and metadata atomically (temporary files + rename) to avoid partial shards.
- Validate the just‑written shard against the schema before returning.

Hard rules:
- No legacy emit paths.
- No optional fields that change shape/meaning of core arrays.


## format.py
Helpers for the canonical on‑disk representation (e.g., filename conventions, extension pairing JSON↔NPZ, dtype checks). Must not introduce alternative layouts.

## id.py
Identity helpers (e.g., composition of shard_id from deterministic components). Stable across platforms given the same inputs.

## meta.py
ShardMeta definition and load/save helpers. All consumers use this single source to parse and validate metadata.
Minimum required fields (contract level): kind (e.g., demux), temperature_K, identifiers sufficient to enforce the uniqueness rule (replica_id, segment_id, round(temperature_K,3)).

Other fields are governed by schema.py and must not be inferred heuristically.


## pair_builder.py
ML‑oriented utility that builds time‑lagged index pairs within a single shard for representation learning.
PairBuilder(tau_steps) - tau_steps > 0 enforced.
make_pairs(shard) - returns contiguous (t, t+tau) indices; returns an empty set if n_frames <= tau.

Contract:
- Never crosses shard boundaries.
- Does not alter shard contents; produces indices only.

## schema.py
Holds the authoritative validation schema and helpers. All shards must validate here; no other schema sources are permitted.

# Maintenance and evolution
- Changes to the shard structure require a deliberate schema version bump and a migration plan. No auto‑migration.
- Document changes here
- Updated tests