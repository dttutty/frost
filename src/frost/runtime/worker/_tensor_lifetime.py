"""Track batch tensors consumed on the worker's model CUDA stream."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import torch

from frost.datatypes.meta import Meta
from frost.utils.tensor_set import TensorSet


def _record_if_cuda(tensor: torch.Tensor, stream: torch.cuda.Stream) -> None:
    if tensor.is_cuda:
        tensor.record_stream(stream)


def record_batch_stream(meta: Meta, stream: torch.cuda.Stream) -> None:
    """Keep producer-stream allocations alive through asynchronous model use.

    ``Meta`` carries tensors allocated by H2D and dynamic-assembly streams.
    A CUDA event orders their writes before model execution; ``record_stream``
    additionally prevents the caching allocator from recycling their storage
    when Python drops ``Meta`` before the model stream has finished using it.
    """
    seen: set[int] = set()

    def visit(value: Any) -> None:  # noqa: ANN401
        if value is None:
            return
        ident = id(value)
        if ident in seen:
            return
        seen.add(ident)

        if isinstance(value, torch.Tensor):
            _record_if_cuda(value, stream)
        elif isinstance(value, TensorSet):
            visit(value.data)
        elif is_dataclass(value) and not isinstance(value, type):
            for field in fields(value):
                visit(getattr(value, field.name))
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(key)
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(meta)
