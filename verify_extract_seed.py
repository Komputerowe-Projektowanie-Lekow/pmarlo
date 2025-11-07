#!/usr/bin/env python3
"""Quick verification that extract_seed is accessible from pmarlo.api"""

from pmarlo.api import extract_seed

# Test cases
test_configs = [
    {"seeds": {"analysis": 42}},
    {"seeds": {"global": 100, "shuffle": 200}},
    {"seeds": {"deeptica": 999}},
    {},
    {"seeds": "invalid"},
]

print("Testing extract_seed function from pmarlo.api:")
print("=" * 50)

for i, config in enumerate(test_configs, 1):
    result = extract_seed(config)
    print(f"Test {i}: {config}")
    print(f"  Result: {result}")
    print()

print("✓ All tests completed successfully!")
print("✓ extract_seed is now available in pmarlo.api")

