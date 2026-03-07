import atexit
import logging
import signal
import sys
from multiprocessing import resource_tracker, shared_memory

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
    if rtype == "shared_memory":
        return  # Skip registering shared memory so the resource tracker ignores it
    return resource_tracker._resource_tracker.register(name, rtype)


def _patched_unregister(name, rtype):
    if rtype == "shared_memory":
        return  # Nothing was registered, so nothing to unregister
    return resource_tracker._resource_tracker.unregister(name, rtype)


resource_tracker.register = _patched_register
resource_tracker.unregister = _patched_unregister

if hasattr(resource_tracker, "_resource_tracker"):
    # Also patch the internal singleton object's methods (required for Python 3.10)
    if resource_tracker._resource_tracker is not None:
        resource_tracker._resource_tracker.register = _patched_register
        resource_tracker._resource_tracker.unregister = _patched_unregister

_TRACKED_SHM_OBJECTS: dict[str, shared_memory.SharedMemory] = {}
_IS_HOST_PROCESS = False
_CLEANUP_DONE = False


def _normalize_shm_name(name: str) -> str:
    if name.startswith(_SHM_NAME_PREFIX):
        return name
    if name.startswith("__"):
        return f"{_SHM_NAME_PREFIX}{name}"
    return f"{_SHM_NAME_PREFIX}_{name}"


def _signal_handler(signum, frame):
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
