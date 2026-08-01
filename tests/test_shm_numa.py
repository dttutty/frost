from __future__ import annotations

import ctypes

from frost.shm import _core


def test_mbind_builds_multiword_node_mask(monkeypatch) -> None:
    calls: list[tuple[int, int, int, list[int], int, int]] = []

    class FakeMbind:
        def __call__(self, address, length, mode, mask, maxnode, flags):
            calls.append(
                (
                    address.value,
                    length.value,
                    mode,
                    list(mask),
                    maxnode.value,
                    flags,
                )
            )
            return 0

    class FakeNuma:
        mbind = FakeMbind()

    monkeypatch.setattr(_core, "_load_numa_library", lambda: FakeNuma())

    _core._mbind_address_to_node(0x1000, 8192, 65)

    bits_per_word = ctypes.sizeof(ctypes.c_ulong) * 8
    assert calls == [
        (0x1000, 8192, _core._MPOL_BIND, [0, 2], bits_per_word * 2, 0)
    ]


def test_region_bind_failure_falls_back_to_first_touch(monkeypatch) -> None:
    class FakeShm:
        buf = memoryview(bytearray(4096))

    def fail_bind(_address: int, _length: int, _node: int) -> None:
        raise OSError("not permitted")

    monkeypatch.setattr(_core, "_mbind_address_to_node", fail_bind)
    _core._NUMA_BIND_WARNED_NODES.discard(1)

    applied = _core._bind_shared_memory_before_touch(
        FakeShm(),  # type: ignore[arg-type]
        length=4096,
        node=1,
    )

    assert not applied
    assert 1 in _core._NUMA_BIND_WARNED_NODES
