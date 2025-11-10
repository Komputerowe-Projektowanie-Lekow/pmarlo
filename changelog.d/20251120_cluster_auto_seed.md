## fixed

- Ensured `_create_clustering_estimator` now detects whether the selected estimator accepts `fixed_seed` at construction so deterministic seeding can be applied in the same way for real deeptime estimators and lightweight mock replacements, preventing the auto-selection sampling test from crashing with unexpected keyword errors while keeping reproducible clustering for real runs (`src/pmarlo/markov_state_model/clustering.py`).
