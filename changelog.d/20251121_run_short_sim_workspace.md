fixed:
- Restore the workspace helper so `run_short_sim` can delegate to the real backend (`run_sampling`)
  while normalizing persisted paths, preventing the previously raised `AttributeError`.
