## fixed

- Align `tests/unit/app/test_app_gating.py` with the actual skip banner source by asserting the `DEEPTICA_SKIP_MESSAGE` constant contains the expected text, ensuring the gating regression is caught directly at the data layer.
