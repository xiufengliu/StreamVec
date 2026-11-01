"""StreamVec: Efficient Incremental Index Maintenance for ANN over Vector Streams.

A faithful pure-Python/NumPy implementation of the StreamVec algorithm: an
HNSW-style layered proximity graph maintained under a mixed insert/delete/update
stream via lazy tombstone deletion, a degree-deficit DirtySet priority queue,
priority-guided repair, and an adaptive rebuild trigger.

See ``README.md`` for the design rationale.
"""

from .index import StreamVecIndex, IndexStats
from .distance import DistanceMetric

__all__ = ["StreamVecIndex", "IndexStats", "DistanceMetric"]
__version__ = "0.1.0"
