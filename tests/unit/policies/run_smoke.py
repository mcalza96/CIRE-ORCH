
import sys
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

from tests.unit.policies.test_policies_smoke import (
    test_query_splitter_basic,
    test_query_splitter_single,
    test_scope_policy_missing,
    test_retry_policy_relax
)

if __name__ == "__main__":
    try:
        test_query_splitter_basic()
        print("test_query_splitter_basic PASS")
        test_query_splitter_single()
        print("test_query_splitter_single PASS")
        test_scope_policy_missing()
        print("test_scope_policy_missing PASS")
        test_retry_policy_relax()
        print("test_retry_policy_relax PASS")
        print("ALL SMOKE TESTS PASSED")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAIL: {e}")
        exit(1)
