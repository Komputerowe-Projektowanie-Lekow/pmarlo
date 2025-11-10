## fixed

- Ensure `demultiplex_run` validates the temperature ladder size against the replica trajectories before reading the exchange log so ladder mismatches surface immediately rather than being masked by missing files.
