"""Select the immutable LocBench runtime before importing the harness packages."""

import os

ROBUST_PROFILE = "robust-null-v3"

if os.environ.get("LOC_COLLECTION_PROFILE") == ROBUST_PROFILE:
    from runtime_v3 import activate

    activate()
    from runtime_v3 import *  # noqa: F403
else:
    from runtime_v2 import *  # noqa: F403
