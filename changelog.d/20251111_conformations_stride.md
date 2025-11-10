## fixed

- Conformations analysis now infers and enforces the physical frame stride recorded in shard metadata, so strided shards whose `source.range` spans more frames than the feature count no longer fail with `frame range ... does not match feature count` and representative extraction resolves the correct trajectory indices for those frames.
- The `tests/unit/io/test_iterload_stream.py::test_iterload_streaming` fixture now supplies a dedicated temporary `output_dir` so `EnhancedMSM` can initialize without raising `MSMBase requires \`output_dir\` to be provided.` and the log assertions stay focused on streaming behavior.
