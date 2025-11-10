## fixed

- Align the aggregate-and-build integration helpers with canonical shard IDs so the generated shards now include the recorded `run_id`, allowing `write_shard` to accept them and the bundle workflow tests to run without value errors.
