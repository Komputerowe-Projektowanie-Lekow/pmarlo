"""Simple test to verify the timestamp function is accessible from pmarlo.api."""

from pmarlo.api import timestamp


def test_timestamp_import():
    """Test that timestamp can be imported from pmarlo.api."""
    ts = timestamp()
    print(f"✓ Successfully imported timestamp from pmarlo.api")
    print(f"  Generated timestamp: {ts}")

    # Verify format
    assert len(ts) == 15, f"Expected length 15, got {len(ts)}"
    assert ts[8] == '-', f"Expected '-' at position 8, got '{ts[8]}'"
    print(f"✓ Timestamp format is correct (YYYYMMDD-HHMMSS)")


def test_webapp_import():
    """Test that webapp can still use _timestamp from utils."""
    from pmarlo_webapp.app.backend.utils import _timestamp

    ts = _timestamp()
    print(f"✓ Successfully imported _timestamp from webapp utils")
    print(f"  Generated timestamp: {ts}")

    # Verify format
    assert len(ts) == 15, f"Expected length 15, got {len(ts)}"
    assert ts[8] == '-', f"Expected '-' at position 8, got '{ts[8]}'"
    print(f"✓ Webapp backward compatibility maintained")


if __name__ == "__main__":
    print("Testing timestamp function migration...\n")

    test_timestamp_import()
    print()
    test_webapp_import()

    print("\n✓ All tests passed! The timestamp function is now available in pmarlo.api")
    print("  and the webapp maintains backward compatibility.")

