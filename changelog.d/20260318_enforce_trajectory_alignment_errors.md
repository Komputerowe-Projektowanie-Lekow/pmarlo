## Fixed

- Raised explicit errors when trajectory alignment fails instead of silently
  returning unaligned data, ensuring data issues are detected early in the
  workflow by `_align_trajectory`.
