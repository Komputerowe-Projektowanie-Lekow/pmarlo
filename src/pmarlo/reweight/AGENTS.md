Reweight module:
- unnegotiable
- explanation
- key values
- best practices
- wanted outcomes
- examples
- files description
- maintenance and evolution

# Unnegotiable
- Deterministic behavior: Given the same input shards, reference temperature, and configuration, reweighting must always produce the identical output weights (bit-for-bit identical arrays). No randomness or run-to-run variation is allowed. Reproducibility is enforced by using a fixed reference temperature (Reweighter(temperature_ref_K)) and recording any relevant parameters/digests.
- No silent fallbacks: If any required thermodynamic data (energy or bias) is missing for a shard, the module must not silently default to uniform weights. Instead, it should emit a clear error message and halt processing. The only acceptable “no data” outcome is an explicit failure (“no weights” state) that calling code can detect and handle (e.g., by treating it as uniform after logging).
- Input immutability: Input shards and datasets must never be mutated in-place. Shards are the canonical immutable records of simulation data. Reweighting should emit new weight arrays as sidecar outputs rather than modifying existing NPZ shards. For example, write weights under a separate mapping (e.g. a {shard_id: w_frame} map or sidecar file) or attach them only to a copy. Do not rewrite or alter the original shard files.
- Consistent key naming: Use a single agreed key for all weight arrays. In particular, the reweighting output array for each frame should be placed under the key w_frame (both in any NPZ sidecar and in-memory structures). Downstream code (MSM builders, training pipelines) will look for this w_frame key consistently.
- Single-kinded shards: Only one shard “kind” (ensemble) may be processed at a time. Input datasets must consist of shards of the same type (e.g. all “demux” shards). Mixed kinds (e.g. demux + replica) or data crossing shard boundaries are forbidden. Each shard contains data from exactly one temperature and segment; reweighting computations must not combine or mix data across shards.
- Explicit temperature: Every split (shard) must explicitly define its simulation temperature (via beta or temperature_K). If a split lacks this, the reweighter should immediately error out. The reference temperature T_ref must also be explicitly specified by the user when constructing the Reweighter.
- MBAR/TRAM only (current scope): Reweighting is currently limited to single-ensemble MBAR (via mode="MBAR") and a placeholder mode="TRAM". In the present implementation, TRAM is treated identically to MBAR internally (it shares the same estimator). The API is intentionally narrow; other estimators are not supported now but may be plugged in later without changing the existing interface.

# Small explanation
The pmarlo.reweight module computes per-frame statistical weights that transform data from its simulation temperature to a common reference temperature $T_\text{ref}$. This is done after shard emission and before clustering/MSM/FES analysis. Users construct a Reweighter with a chosen temperature_ref_K (the reference $T_\text{ref}$). Then calling Reweighter.apply(dataset, mode=<MBAR/TRAM>) processes the dataset’s shards (splits) and produces weight arrays.

The input data contract is: dataset must be a mapping containing a 'splits' dictionary (one entry per shard). Each split entry must provide either beta or temperature_K metadata, and may include:
- energy: a 1D array of potential energy values (kJ/mol) per frame.
- bias: a 1D array of bias potentials (if any) per frame.
- weights: an existing per-frame weight array from previous steps (to be multiplied).
If beta or temperature_K is missing for a split, Reweighter.apply raises an error.

Once the splits are validated, reweighting proceeds deterministically for each split (shard). For each split, the (currently MBAR-style single-ensemble) algorithm is:

- Compute the number of frames $N$.
- Fail fast on missing thermodynamics: If the shard has no energy (and no bias), the operation must abort (no silent uniform weights).
- Otherwise, compute unnormalized weights for each frame: w_i is in exp(-(B_ref - B_sim)E_i - B_ref*B_i)
- Normalize the weights so they sum to 1.0. If the sum is zero or non-finite (indicating numerical issues), handle it as a failure (emit an error) rather than hiding the issue.
- Store the resulting normalized weights array (dtype float64) for that split.

These weights are attached back to the dataset but only as new data. Specifically, after computing each split’s weight array w_frame, the module:

Inserts it into dataset["splits"][split_name]["weights"] and updates a global dataset["frame_weights"][split_name] mapping (if the dataset is a mutable mapping).

Caches it internally in Reweighter._cache[shard_id] so that reweighting the same shard again (with same ref temperature) does not recompute the weights.

In the full workflow, reweighting “sits” right after shards are loaded and optional CV (deep TICA/VAMP) training, and immediately before clustering/MSM/FES. A typical pipeline is: DEMUX simulation data → produce demux shards → (optional) use w_frame as sample weights in CV training → then apply reweighting to $T_\text{ref}$ → feed results into clustering, MSM building, free-energy surface estimation, etc. The final MSM/FES analyses always use the frame weights at $T_\text{ref}$, and the reference temperature must be made explicit in configs and filenames.

# Key values
- Reproducibility: Every run must be repeatable. The reweighting output is a pure function of the input shards and T_ref (and mode); identical inputs yield identical outputs. Keep any random seeds or config versions recorded for traceability.
- Immutability: Shards (data splits) are authoritative and immutable. The reweight module adds new weight arrays but never alters existing arrays/metadata.
- Transparency and Observability: The process must fail fast on errors (missing data, invalid inputs) and log a clear reason. No hidden defaults. The schema of inputs is strictly enforced (e.g., presence of temperature/Beta) so that downstream code can rely on it.
- Consistency: Use one canonical format. All weights use the same key (w_frame) and units. The API surface is intentionally small (only AnalysisReweightMode enum and Reweighter class) to avoid confusion.
- Single-responsibility: The module only computes statistical reweighting. It does not modify shards, train models, or perform clustering. Upstream modules (demux, shard management) and downstream modules (MSM, FES) have separate concerns.
- Fail-fast safety: If any invariant is violated (e.g. unsupported mixed shards, missing thermodynamics), the module should raise an error or exit explicitly, rather than produce silently incorrect weights.
- Scientific Correctness: Frame weights must accurately reflect the Boltzmann reweighting formula at $T_\text{ref}$. Do not approximate by “close enough” heuristics. The math is precise and must be applied as stated.


# Best practices
1. Validate inputs early. Before computing weights, ensure each split’s metadata passes validation (exact match on temperature_K or valid beta, correct array lengths, etc.). This should mirror shards-schema validation. Fail promptly on any missing or malformed field.
2. Explicit T_ref units. Always specify Reweighter(temperature_ref_K=...) in Kelvin, and treat it as an exact filter in any selection. Document in logs or config that all weights are relative to this temperature.
3. Cache semantics. If reweighting is called multiple times in a run, repeated splits (same shard_id) will reuse cached weights. This cache assumes the shard’s thermodynamic arrays (energy/bias) have not changed. Clearing the cache requires creating a new Reweighter instance.
4. Consistent key usage. Store weights arrays under w_frame. When writing NPZ sidecars for shards, use w_frame (not some alternate key). In-memory, the convenience dictionary should also use w_frame entries. This ensures downstream tools always find weights by the same key.
5.No mixed batches. Feed only shards of a single kind (ensemble) into Reweighter.apply at once. Partition mixed inputs ahead of time. The code assumes each shard’s data stand alone and will throw or misbehave if you mix “replica” and “demux” shards, or multiple temperatures in one split.
6. Logging on fallback. If any split lacks energy/bias (and thus cannot be reweighted), do not quietly continue. Log an error including the shard ID and reason, and skip or abort as decided by policy. (The trainer may interpret missing weights as uniform after a logged failure.)
8/ Versioning and defaults. The mode enum AnalysisReweightMode may be extended in future. Freeze the meaning of "MBAR" and "TRAM" now: "TRAM" is currently mapped to MBAR internally. If adding new modes or changing behavior, bump the module version and update documentation accordingly. Avoid changing defaults silently.

# Wanted outcomes
- Accurate frame weights: After reweighting, each shard has a w_frame array of non-negative floats summing to 1.0 (within numerical tolerance). These weights correctly reweight the Boltzmann distribution from the simulation temperature to T_ref.
- No corruption of shards: The original shard files (metadata and arrays) remain unchanged. All additional data (weights) is placed in separate mappings or new files. The function never alters pre-existing fields in the shard.
- Predictable pipeline behavior: Downstream modules (MSM building, FES, CV training) can reliably find per-frame weights under the same keys (w_frame or via dataset["frame_weights"]) for every shard, if requested. Optional use of weights in training is transparent: if weights are missing, training proceeds with uniform weights but only after an explicit note in logs.
- Clear failures: If a split is missing required data or fails sanity checks, the job should fail (or skip that split) with a clear message, so engineers can investigate. There should be no silent surprises (like inadvertently uniform distributions without warning).
- Extensible design: The module’s API should remain stable so that future improvements (e.g. real multi-ensemble TRAM, alternative estimators) can be plugged in without altering the usage pattern. This means new logic can be added internally, but the interface (Reweighter.apply, weight keys, error codes) stays the same.
- Consistent reproducibility: Running the same analysis pipeline (with the same shards and ref temperature) on different machines or at different times yields identical results. This is achieved by recording the reference temperature and any RNG seeds (if used) and using deterministic algorithms.

# Examples
1. Deterministic behavior. Given the same input shards and T_ref, calling Reweighter.apply(dataset, mode="MBAR") twice should yield byte-for-byte identical w_frame arrays. Similarly, setting mode="TRAM" (currently an alias for MBAR) must also produce the same deterministic output.
2. Missing energy or bias. If a shard split lacks both energy and bias arrays, the reweighter should NOT silently return uniform weights. Instead, it logs an error (e.g. "No energy/bias for shard X – cannot compute weights") and stops processing. Downstream code might then treat this as an explicit "no weights" case (equivalent to all weights equal) but only after the failure is recorded.
3. Immutability of input. If dataset is backed by NPZ shard files, reweighting must leave them untouched. For example, do not rewrite the NPZ to insert weights. Instead, write a separate "weights sidecar" or return a shard_id → w_frame mapping so that another step can attach the weights later.
4. Consistent w_frame key. All code (file loaders, analysis tools) should expect the reweighted weights under a key named w_frame, not, say, weights or frame_weights. The Reweighter.apply() method will set split["weights"] in the dataset for convenience, but we treat that as an internal choice. Externally, the agreed key for output is w_frame.
5. Single ensemble input. Suppose you have shards from two different simulation ensembles (e.g. temperatures 300K and 350K, or two different Hamiltonians). You must call the reweighter separately for each ensemble. Do not pass them together, as mixing kinds violates an invariant.
6. Use in training vs analysis. During model training (e.g. DeepTICA), you may optionally feed w_frame as sample_weight. If w_frame is absent, training should behave identically as if weights were uniform – but note in documentation that “training without provided weights” occurred. In final analysis, however, reweighting is mandatory: before clustering/MSM/FES we must always have applied reweighting to the common T_ref.
7. Future estimators. Right now only MBAR (and TRAM-as-MBAR) are implemented. If in the future a new estimator (e.g. a different multi-ensemble algorithm) is added, it should be plugged into the Reweighter interface without changing how callers use it. The mode enum allows addition of new names, but existing behavior for "MBAR" and "TRAM" remains fixed.

# Files description

## reweighter.py
AnalysisReweightMode: An enum ("none", "MBAR", "TRAM") selecting the reweighting algorithm. Input is normalized (case-insensitive). Currently "TRAM" maps to the MBAR estimator internally.

Reweighter: Main class that performs reweighting.

Constructor: Reweighter(temperature_ref_K) requires a positive finite reference temperature (Kelvin). It computes and stores $\beta_{\text{ref}}=1/(k_B T_\text{ref})`.
