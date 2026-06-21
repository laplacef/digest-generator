"""Concrete content sources: packages that fetch external data into ``Entry``.

Each sub-package implements one source (currently ``rss/``). Sources
share two contracts: every fetcher accepts a ``Filter`` (date range,
limit) and produces ``digest_generator.core.types.Entry`` objects. A
``ContentSource`` Protocol that ``api.fetch`` could iterate over is
anticipated by this layer's consistent shape but remains undefined
until a second concrete source exists to define it against.
"""
