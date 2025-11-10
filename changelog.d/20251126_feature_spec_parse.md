## fixed

- Ensure molecular feature specs like `distance([0, 1])` emit only the supported `indices` kwargs so downstream consumers can compare expectations without extra keys.
- Keep the `phi_psi` built-in wrapped to (-π, π] so no column ever reports -180° and the Ramachandran wiring assertions stay valid.
