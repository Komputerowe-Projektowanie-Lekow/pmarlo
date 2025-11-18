### fixed
- Hardened `_find_transition_states` by validating temperatures, populations, committors, flux vectors, KIS scores, and source/sink indices before building transition conformations.

### added
- Behavioral test suite for `_find_transition_states` that covers committor-based labeling, energetic calculations, metadata propagation, and defensive error handling.
