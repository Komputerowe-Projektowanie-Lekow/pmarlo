## fixed

- Flush the fast DCD writer after every header update so the crash-recovery snapshot is fully persisted even if the process is terminated before closing the file, which keeps Windows readers happy.
- The crash-recovery regression test now deletes the reporter reference and runs `gc.collect()` before loading the DCD, mirroring the OS closing handles after a crash and avoiding Windows sharing errors.
