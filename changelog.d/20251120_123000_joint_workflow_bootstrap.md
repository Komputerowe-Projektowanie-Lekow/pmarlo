### added
- Joint workflow bootstraps a CV transform from reference-temperature shards using weighted whitening and pair-derived VAMP-2 seeding, recording training history for guardrail checks.
- Iteration metrics now derive from actual CV outputs, clustering, ITS, and Chapman-Kolmogorov diagnostics instead of placeholder values.

### changed
- CV bootstrap and iteration now raise explicit errors when shards or lagged pairs are missing to avoid silent fallbacks, and VAMP-2 history is consolidated to prevent duplicate guardrail readings.
