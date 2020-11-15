"""Module that implements classes that use ray."""
import os
import sys

N_GPUS_ACTOR = float(os.getenv("N_GPUS_ACTOR", 0))
MAX_CALLS_ACTOR = int(os.getenv("MAX_CALLS_ACTOR", 1e20))

try:
    import ray
except ImportError as e:
    if sys.version_info <= (3, 7):
        raise e
    else:

        class ray:
            """Dummy to avoid import errors before ray is released for Python 3.8."""

            def __getattr__(self, item):
                return lambda *args, **kwargs: None

            def remote(self, *args, **kwargs):
                """Do nothing."""
                pass
