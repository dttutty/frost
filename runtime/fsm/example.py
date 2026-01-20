from collections import deque
from dataclasses import dataclass, field

import torch

from schemas.types.fsm import StepResult, WorkerState


@dataclass(slots=True)
class WorkerContext:
    model: torch.nn.Module
    optim: torch.optim.Optimizer
    loader: iter
    device: torch.device
    events: deque = field(default_factory=deque)

    # runtime fields
    batch_cpu: object | None = None
    batch_gpu: object | None = None
    loss: torch.Tensor | None = None
    step: int = 0
    save_requested: bool = False
    stop_requested: bool = False
    error: Exception | None = None


class WorkerFSM:
    def __init__(self, ctx: WorkerContext):
        self.ctx = ctx
        self.state = WorkerState.IDLE

    def step(self) -> StepResult:
        try:
            if self.state == WorkerState.IDLE:
                return StepResult(WorkerState.PREFETCH)

            if self.state == WorkerState.PREFETCH:
                return self._prefetch()

            if self.state == WorkerState.COMPUTE:
                return self._compute()

            if self.state == WorkerState.APPLY_UPDATE:
                return self._apply_update()

            if self.state == WorkerState.SAVE:
                return self._save()

            if self.state == WorkerState.STOPPED:
                return StepResult(WorkerState.STOPPED)

            raise RuntimeError(f"Unknown state: {self.state}")

        except Exception as e:
            self.ctx.error = e
            return StepResult(WorkerState.STOPPED)

    def _prefetch(self) -> StepResult:
        ctx = self.ctx
        if ctx.stop_requested:
            return StepResult(WorkerState.STOPPED)

        # fetch CPU batch
        try:
            ctx.batch_cpu = next(ctx.loader)
        except StopIteration:
            ctx.stop_requested = True
            return StepResult(WorkerState.STOPPED)

        # move to GPU
        ctx.batch_gpu = move_to_device(ctx.batch_cpu, ctx.device, non_blocking=True)
        return StepResult(WorkerState.COMPUTE)

    def _compute(self) -> StepResult:
        ctx = self.ctx
        ctx.model.train()

        out = ctx.model(ctx.batch_gpu)
        loss = out["loss"] if isinstance(out, dict) else out
        ctx.loss = loss

        loss.backward()
        return StepResult(WorkerState.APPLY_UPDATE)

    def _apply_update(self) -> StepResult:
        ctx = self.ctx
        ctx.optim.step()
        ctx.optim.zero_grad(set_to_none=True)
        ctx.step += 1

        # 外部事件：例如每 N step 保存一次
        if ctx.save_requested:
            return StepResult(WorkerState.SAVE)

        return StepResult(WorkerState.PREFETCH)

    def _save(self) -> StepResult:
        ctx = self.ctx
        save_checkpoint(ctx.model, ctx.optim, ctx.step)
        ctx.save_requested = False
        return StepResult(WorkerState.PREFETCH)
