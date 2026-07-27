from frost.protocols.analyzer_proto import AnalyzerProto as LegacyAnalyzerProto
from frost.runtime.host import AnalyzerProto


def test_analyzer_protocol_is_host_owned_with_legacy_compatibility() -> None:
    assert LegacyAnalyzerProto is AnalyzerProto
