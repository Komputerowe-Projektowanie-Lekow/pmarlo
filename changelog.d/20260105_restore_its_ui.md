### Fixed
- Restored the Streamlit Implied Timescales tab so shard selections, configuration, and plotting now call `calculate_its` again instead of rendering an empty panel.
- Brought back the Model Preview and Assets tabs with full bundle inspection, workspace inventories, and one-click loading into the corresponding workflow stages.
- Resolved Streamlit session-state assignment errors by routing ITS form updates through pending keys before widgets instantiate.

### Added
- Added helper utilities to infer default topology/feature spec paths and tabularize implied timescale outputs for the example app workflow.
- Introduced reusable table formatters for runs, shards, models, analyses, and conformations so the Assets tab surfaces actionable metadata instead of empty placeholders.
