from schemas.types.fsm import WorkerEvent, WorkerState


class WorkerFSM:
    def __init__(self, ctx):
        self.ctx = ctx
        self.state = WorkerState.IDLE

    def step(self, event: WorkerEvent) -> None:
        s = self.state

        if s == WorkerState.IDLE:
            if event == WorkerEvent.META_READY:
                self.state = WorkerState.PREFETCH
                return

        if s == WorkerState.PREFETCH:
            self.ctx.prefetch()
            self.state = WorkerState.COMPUTE
            return

        if s == WorkerState.COMPUTE:
            self.ctx.compute()
            self.state = WorkerState.APPLY_UPDATE
            return

        if s == WorkerState.APPLY_UPDATE:
            self.ctx.apply_update()
            self.state = WorkerState.WAIT_META
            return

        if event == WorkerEvent.SAVE_MODEL:
            self.ctx.save_model()
            return

        if event == WorkerEvent.STOP:
            self.state = WorkerState.STOPPED
            self.ctx.stop_all()
            return

        raise RuntimeError(f"invalid transition: {s} + {event}")
