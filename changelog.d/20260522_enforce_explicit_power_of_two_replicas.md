### Removed
- Removed the automatic ~5 K spacing fallback in `pmarlo.utils.replica_utils.power_of_two_temperature_ladder`; callers must now provide an explicit replica count and invalid degenerate ladders raise immediately.
