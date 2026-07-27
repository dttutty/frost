"""Compatibility import for the host-owned analyzer protocol.

New code should import :class:`AnalyzerProto` from
``frost.runtime.host.protocols``.
"""

from frost.runtime.host.protocols import AnalyzerProto

__all__ = ["AnalyzerProto"]
