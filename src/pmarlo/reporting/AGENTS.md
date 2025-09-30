Reporting module:
- unnegotiable
- explanation
- key values
- best practices
- wanted outcomes
- examples
- files description
- maintenance and evolution

# Unnegotiable
- Deterministic Outputs: All CSV, JSON, and PNG files must be identical on repeated runs given the same inputs/environment. Use fixed seeds (if any randomness is involved), constant figure sizes, DPI, and fonts to ensure bit-for-bit reproducibility. No dynamic content (timestamps, UUIDs, dates) may appear in file names or file contents.
- Strict Naming Scheme: Follow the prescribed artifact naming pattern exactly: kind__Tref=<K>__tau=<steps>__data=<short_digest>__cfg=<short_digest>__v<semver>.(csv|json|png). Do not use generic names like “latest” or include timestamps. The kind (e.g. “states” or “fes_contour”), reference temperature (Tref), lag time (tau), data/config digests, and semantic version must all be present. Names must be deterministic and collision-free, typically by hashing input data or configs to produce <short_digest>.
- Canonical Directory Layout: Write all outputs under a fixed, well-known directory structure. The module itself must create any needed directories (e.g. via Path(output_dir).mkdir(parents=True, exist_ok=True)). Do not scatter files unpredictably; e.g. put all output artifacts for a run under one root directory (like output_dir/) or documented subfolders.
- Fail-Fast on Bad Input: Validate all inputs before proceeding. If any required input array is empty, shape-mismatched, or contains NaNs/infinities, raise an error immediately (for example, fes2d already raises a ValueError on empty or mismatched inputs). Do not attempt to guess or fallback silently. There are no “soft” failure modes – every anomaly should produce a clear exception.
- Static Format Only: Only generate static, non-interactive outputs. Accepted formats are PNG (for plots), CSV, and JSON. Do not produce HTML, PDF, Jupyter notebooks, or other interactive/complex formats in this module. Ensure that every output is a plain file suitable for archival and comparison.

# Small explanation
The reporting module runs after the MSM/FES analysis step to produce human-readable summaries and visualizations of the results. It provides functions to export MSM data (states, transition matrices, etc.) and free-energy surfaces to disk. The public API consists of five functions: write_conformations_csv_json, fes2d, save_fes_contour, save_pmf_line, and save_transition_matrix_heatmap. Together, these functions take arrays or summaries from the analysis and write out standardized CSV/JSON tables and PNG plots. All outputs must adhere to the reproducibility and naming rules outlined above, fitting into the deterministic PMARLO workflow. 

# Key values
- Reproducibility: Ensure that every output file is reproducible on every run. This means controlling all sources of non-determinism (random number generators, system locales, font rendering) so that identical inputs produce identical outputs. For example, plotting functions fix figsize and dpi (e.g. plt.figure(figsize=(6,5)) and plt.savefig(..., dpi=200)) to avoid variation in image output.
- Transparency: Output data should be explicit and self-describing. Any metadata (like temperature, time step, etc.) must appear in filenames or within the content (CSV/JSON) so users can trace the provenance. Avoid hidden magic – do not rely on external context to interpret a file.
- Determinism: No stochastic or environment-dependent behavior in reporting. Do not sample or subsample data. Use fixed colormaps (e.g. “viridis”), fixed figure layouts (tight_layout is used in code to ensure consistent spacing), and avoid any code that could introduce randomness in the final output.
- Robustness: Strictly enforce input validation. The module should catch invalid inputs early (shape mismatches, missing data, NaNs) and raise errors rather than produce misleading output. For example, fes2d checks for empty or mismatched x,y arrays and raises a ValueError if so. Continue this fail-fast philosophy for all functions.
- Consistency: Maintain consistent structure across outputs. All CSV/JSON exports should use the same formatting conventions (e.g. sorted keys, UTF-8 encoding). As shown in write_conformations_csv_json, the CSV header fields are sorted and written uniformly. Similarly, JSON dumps use a fixed indent level. Plots should have consistent style (labels, titles, colorbars) so that different plots look cohesive.

# Best practices
- Directory Handling: Always create the target directory before writing. Use code like Path(output_dir).mkdir(parents=True, exist_ok=True) so that repeated runs succeed without manual setup. Never write files outside the designated output directory.
- Input Sanitization: Before writing outputs, clean or validate data types. The CSV/JSON writer should convert NumPy scalars/arrays to native Python types (as done in _normalize_for_json_row) so that JSON serialization never fails. Check for and reject NaN/Inf values explicitly.
- Naming Files: Apply the canonical naming convention exactly. For example, if saving a heatmap of the transition matrix at 300K, 1000 steps, you might name it heatmap__Tref=300__tau=1000__data=abc123__cfg=def456__v0.1.0.png. Ensure <short_digest> values are deterministic (e.g. take first 8 characters of a SHA256 hash of the underlying data/config). Include the semantic version (vX.Y.Z) of PMARLO to bind the output to a specific code version. Do not vary the order or format of these parts.
- Figure Settings: Explicitly set all figure parameters. E.g. use plt.figure(figsize=(7,6)) (as in save_fes_contour) and a fixed dpi. Lock in a known font (e.g. via matplotlib.rcParams) so that labels render the same on any machine. Call plt.tight_layout() to avoid cropping differences. Always call plt.close() after saving to free memory.
- Return Values: When possible, have plotting functions return the file path of the created image (the code returns a str path) so callers can record or log where files went. Consistently return None or an empty string if writing fails, but ideally this never happens (and should be treated as an exception condition).
- Metadata Inclusion: If relevant, embed essential metadata inside the CSV/JSON (not just in file names). For example, the CSV of conformations could include columns like temperature_K or n_frames if available (similar to how shard JSON includes schema version, temperature, etc. in format.py). Ensure that such fields are present so the CSV is self-contained.

# Wanted outcomes
- Fixed Set of Artifacts: Each run of the reporting module produces a predictable collection of files. For example: one CSV and one JSON of conformation summaries, one PNG FES contour plot, one PNG PMF line plot, and one PNG transition-matrix heatmap. No extra or missing files.
- Bit-for-Bit Reproducibility: Rerunning the same analysis yields identical outputs (same bytes, same image pixels). This means versioning (the vX.Y.Z tag) and data hashes are critical. Users should be able to diff the new outputs against old ones and see zero differences when inputs/configs haven’t changed.
- Clarity and Quality: All PNG plots are clear and correctly labeled. Axes should have appropriate units (e.g. free energy in kJ/mol), and titles should reflect the data (as seen in save_fes_contour and save_pmf_line). Colorbars and legends should be legible.
- No Hidden Side Effects: The module must not depend on any mutable global state. It should not, for instance, set a global random seed for someone else’s code. All randomness (if any, e.g. in hexbin sampling) should be seeded locally or disabled. The example code in save_fes_contour avoids unpredictable behavior by using a fixed gridsize=40 in hexbin.
- Deterministic Reports: Even if the input data distribution is sparse (many zero or empty bins), the module should not silently degrade output quality. If a plot can’t be meaningfully rendered (e.g. >60% empty FES bins), the function may raise an informative error or warning. It should never produce a misleading or blank image without notifying the user (the current code does a fallback hexbin or contour, but per policy this case should probably error out instead to maintain strictness).


# Examples
1. ... TODO(need to find the issues and find the resolutions)

# Files description

## export.py 
Implements write_conformations_csv_json which creates the output directory if needed and writes a CSV and JSON of the given item list. It normalizes any NumPy types so that the CSV and JSON have native Python types.

## plots.py
Contains all plotting routines:
- save_transition_matrix_heatmap(T, output_dir, name): makes a 6×5″ PNG of the matrix T using imshow.
- save_fes_contour(F, xedges, yedges, xlabel, ylabel, output_dir, filename, mask): makes a contour (or hexbin fallback) plot of 2D free energy F.
- save_pmf_line(F, edges, xlabel, output_dir, filename): plots a 1D PMF line from F vs bin centers.
- fes2d(x, y, bins, adaptive, temperature, min_count): computes and returns a 2D free energy surface (in kJ/mol) and bin edges; it raises a ValueError on invalid input

## __init__.py
Exports the public API (write_conformations_csv_json, fes2d, save_fes_contour, save_pmf_line, save_transition_matrix_heatmap)

## Unit Tests
Located under tests/unit/reporting/, these verify that the above functions create the expected files (e.g. test_export.py checks that the CSV/JSON files exist, and test_plots.py checks that PNGs are created). Use these as regression checks to ensure nothing breaks.

# Maintenance and evolution
- Updating the Spec: Whenever you add a new output type or change the file-naming rules, update this document immediately. The naming convention is critical – if the pipeline evolves to include new parameters, expand the pattern (e.g. adding a new __param=value segment). Keep the pattern parsable by scripts.
- Version Bumps: Since the semantic version (vX.Y.Z) appears in file names, bump the version in step with releases. Any incompatible change in output format or naming should be a major version bump. Document the change in both code and here.
- Consistent Behavior: If upgrading dependencies (e.g. Matplotlib), verify that default fonts or rendering haven’t changed in a way that could alter outputs. If they have, adjust defaults in code (pin the font, etc.) or update tests to enforce consistency.
- Regression Checks: Use the existing unit tests as a baseline. For any new feature, add tests that check for exact output consistency (e.g. checksum of a PNG or exact CSV contents). Keep a small library of known-good outputs to diff against.
- Alignment with Workflow: The reporting module must stay in sync with upstream analysis steps (MSM, FES). If those modules add new data fields or change formats, ensure reporting can ingest them or gracefully ignore extras. For example, if MSM adds a new metadata field, decide whether it should appear in the output CSV/JSON.
- Documentation: Update docstrings and this AGENTS.md together. This file is the single source of truth for how reporting should work. Any deviations or additional conventions (e.g. a new required column in CSV) must be reflected here.