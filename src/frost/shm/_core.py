"""Core shared-memory utilities for the FROST framework.

This module provides helpers to allocate and attach POSIX shared-memory
segments as PyTorch tensors, patching the Python resource-tracker so that
shared-memory objects are not prematurely cleaned up when worker processes
exit.  The host process is responsible for physically unlinking segments at
shutdown; workers only close their mappings.
"""
import atexit
import logging
import signal
import sys
from multiprocessing import resource_tracker, shared_memory
from typing import Any, cast

import numpy as np
import torch

logger = logging.getLogger(__name__)

_SHM_NAME_PREFIX = "__FROST"

# Map PyTorch dtypes to NumPy dtypes
_TORCH_TO_NP = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.bool: bool,
}


def _patched_register(name, rtype):
    """Suppress resource-tracker registration for shared-memory segments.

    Patched into :mod:`multiprocessing.resource_tracker` so that FROST-owned
    shared-memory objects are not unlinked when a worker process exits.

    Args:
        name: Resource name passed by the tracker.
        rtype: Resource type string (e.g. ``"shared_memory"``).
    """
    if rtype == "shared_memory":
        return  # Skip registering shared memory so the resource tracker ignores it
    return resource_tracker._resource_tracker.register(name, rtype)


def _patched_unregister(name, rtype):
    """Suppress resource-tracker unregistration for shared-memory segments.

    Counterpart of :func:`_patched_register`; prevents the tracker from
    trying to unlink a segment that was never registered.

    Args:
        name: Resource name passed by the tracker.
        rtype: Resource type string (e.g. ``"shared_memory"``).
    """
    if rtype == "shared_memory":
        return  # Nothing was registered, so nothing to unregister
    return resource_tracker._resource_tracker.unregister(name, rtype)


cast(Any, resource_tracker).register = _patched_register
cast(Any, resource_tracker).unregister = _patched_unregister

if hasattr(resource_tracker, "_resource_tracker"):
    # Also patch the internal singleton object's methods (required for Python 3.10)
    if resource_tracker._resource_tracker is not None:
        cast(Any, resource_tracker._resource_tracker).register = _patched_register
        cast(Any, resource_tracker._resource_tracker).unregister = _patched_unregister

_TRACKED_SHM_OBJECTS: dict[str, shared_memory.SharedMemory] = {}
_IS_HOST_PROCESS = False
_CLEANUP_DONE = False


def _normalize_shm_name(name: str) -> str:
    """Ensure a shared-memory segment name carries the FROST name prefix.

    Args:
        name: Raw segment name, possibly already prefixed.

    Returns:
        Normalised name guaranteed to start with ``_SHM_NAME_PREFIX``.
    """
    if name.startswith(_SHM_NAME_PREFIX):
        return name
    if name.startswith("__"):
        return f"{_SHM_NAME_PREFIX}{name}"
    return f"{_SHM_NAME_PREFIX}_{name}"


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM by cleaning up shared memory before exit.

    Args:
        signum: Signal number received.
        frame: Current stack frame (unused).
    """
    logger.warning(
        f"Process received signal {signum}. Forcing shared memory cleanup..."
    )
    cleanup_all_shm(is_host=_IS_HOST_PROCESS)
    sys.exit(1)


try:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
except ValueError:
    # _core.py imported in a non-main thread; signal registration not possible.
    pass


def set_as_host():
    """Mark the current process as the host (owner) of shared-memory segments.

    The host process calls :func:`cleanup_all_shm` with ``is_host=True`` on
    exit, which physically unlinks (deletes) each segment.  Worker processes
    only close their mappings.
    """
    global _IS_HOST_PROCESS
    _IS_HOST_PROCESS = True


def _auto_cleanup():
    # Registered via atexit; called automatically on process exit
    cleanup_all_shm(is_host=_IS_HOST_PROCESS)


def _unregister_shm(shm_name: str) -> None:
    """
    Hack to prevent resource_tracker from deleting shared memory when a process exits.
    """
    # Try standard name
    try:
        resource_tracker.unregister(shm_name, "shared_memory")
    except Exception:
        pass

    # Fallback for POSIX systems where the tracker appends a leading slash
    try:
        resource_tracker.unregister(f"/{shm_name}", "shared_memory")
    except Exception:
        pass


def cleanup_all_shm(is_host: bool = True) -> None:
    """Close (and optionally unlink) all tracked shared-memory segments.

    Idempotent: subsequent calls after the first are no-ops.

    Args:
        is_host: When ``True`` the segments are physically deleted via
            :meth:`~multiprocessing.shared_memory.SharedMemory.unlink`.
            Worker processes should pass ``False``.
    """
    global _CLEANUP_DONE
    if _CLEANUP_DONE:
        return
    _CLEANUP_DONE = True
    for name, shm in list(_TRACKED_SHM_OBJECTS.items()):
        try:
            shm.close()
            if is_host:
                shm.unlink()  # Physical deletion is only performed by the host process
        except Exception:
            pass
        finally:
            _TRACKED_SHM_OBJECTS.pop(name, None)


def create_shared_mem_array(
    name: str, shape: list | tuple, dtype: torch.dtype
) -> torch.Tensor:
    """Allocate a new POSIX shared-memory segment and return it as a zero-filled tensor.

    If a segment with the given name already exists it is first unlinked so a
    fresh block is created.  A reference to the underlying
    :class:`~multiprocessing.shared_memory.SharedMemory` object is kept in the
    module-level registry to prevent the GC from unmapping the memory.

    Args:
        name: Logical name for the segment (will be normalised with the FROST
            prefix by :func:`_normalize_shm_name`).
        shape: Shape of the resulting tensor.
        dtype: PyTorch dtype; must exist in the ``_TORCH_TO_NP`` mapping.

    Returns:
        A CPU tensor backed by the shared-memory segment, initialised to zero.
    """
    name = _normalize_shm_name(name)
    np_dtype = _TORCH_TO_NP[dtype]
    num_bytes = int(np.prod(shape)) * np.dtype(np_dtype).itemsize

    try:
        # Try to create new shared memory
        shm = shared_memory.SharedMemory(name=name, create=True, size=num_bytes)
    except FileExistsError:
        # Attach to the existing memory and destroy it
        existing_shm = shared_memory.SharedMemory(name=name, create=False)
        existing_shm.unlink()

        # Create a new shared memory block
        shm = shared_memory.SharedMemory(name=name, create=True, size=num_bytes)
    _unregister_shm(shm.name)

    # Keep a reference to the shm object to prevent GC from unmapping the memory
    _TRACKED_SHM_OBJECTS[shm.name] = shm

    np_arr = np.ndarray(shape, dtype=np_dtype, buffer=shm.buf)
    tensor = torch.from_numpy(np_arr)

    # Initialize memory with zeros
    tensor.zero_()
    return tensor


def get_shared_mem_array(
    name: str, shape: list | tuple, dtype: torch.dtype
) -> torch.Tensor:
    """Attach to an existing POSIX shared-memory segment and return it as a tensor.

    The segment must have been created previously by :func:`create_shared_mem_array`.
    A reference to the underlying :class:`~multiprocessing.shared_memory.SharedMemory`
    object is retained to prevent the GC from unmapping the memory.

    Args:
        name: Logical name of the segment (normalised automatically).
        shape: Expected shape of the tensor view.
        dtype: PyTorch dtype matching the data written by the creator.

    Returns:
        A CPU tensor backed by the existing shared-memory segment.
    """
    name = _normalize_shm_name(name)
    np_dtype = _TORCH_TO_NP[dtype]

    shm = shared_memory.SharedMemory(name=name)
    _unregister_shm(shm.name)

    # Keep a reference to the shm object to prevent GC from unmapping the memory
    _TRACKED_SHM_OBJECTS[shm.name] = shm

    np_arr = np.ndarray(shape, dtype=np_dtype, buffer=shm.buf)
    tensor = torch.from_numpy(np_arr)
    return tensor


atexit.register(_auto_cleanup)
