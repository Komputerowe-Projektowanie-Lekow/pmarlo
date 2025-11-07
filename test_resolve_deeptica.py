#!/usr/bin/env python
"""Quick test to verify resolve_deeptica is accessible from pmarlo.api"""

from pmarlo.api import resolve_deeptica

# Test case 1: DeepTICA enabled with parameters
config1 = {
    "deeptica": {
        "enabled": True,
        "min_pairs": "100",
        "skip_on_failure": 1
    }
}
enabled1, params1 = resolve_deeptica(config1)
print(f"Test 1 - Enabled with params:")
print(f"  enabled: {enabled1}")
print(f"  params: {params1}")
print()

# Test case 2: DeepTICA disabled
config2 = {
    "deeptica": {
        "enabled": False,
        "min_pairs": 50
    }
}
enabled2, params2 = resolve_deeptica(config2)
print(f"Test 2 - Disabled:")
print(f"  enabled: {enabled2}")
print(f"  params: {params2}")
print()

# Test case 3: No DeepTICA section
config3 = {
    "other_config": "value"
}
enabled3, params3 = resolve_deeptica(config3)
print(f"Test 3 - No deeptica section:")
print(f"  enabled: {enabled3}")
print(f"  params: {params3}")
print()

# Test case 4: DeepTICA with only enabled key
config4 = {
    "deeptica": {
        "enabled": True
    }
}
enabled4, params4 = resolve_deeptica(config4)
print(f"Test 4 - Only enabled key:")
print(f"  enabled: {enabled4}")
print(f"  params: {params4}")
print()

print("✓ All tests completed successfully!")
print("✓ resolve_deeptica is now available in pmarlo.api")

