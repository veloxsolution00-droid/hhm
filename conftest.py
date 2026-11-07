"""Repository-wide pytest configuration.

The ``backup_*`` directories hold point-in-time snapshots taken before refactors
(for example ``backup_20261105_evidence_aware/test_chat_retrieval.py``). They are
not part of the test suite and encode superseded contracts, so pytest must not
collect them; only the tests under ``tests/`` are the suite.
"""

collect_ignore_glob = ["backup_*/*"]
