from frost.runtime.host import AnalyzerProto
from frost.runtime.host.protocols import AnalyzerProto as HostAnalyzerProto


def test_analyzer_protocol_is_exported_by_host_runtime() -> None:
    assert HostAnalyzerProto is AnalyzerProto
