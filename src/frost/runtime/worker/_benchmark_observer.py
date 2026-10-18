"""Optional benchmark observation for the worker training loop."""

from __future__ import annotations

import json
import time

import torch

from frost.datatypes.env import Env
from frost.runtime.worker._telemetry import (
    _discover_sibling_rank_pids,
    _get_bool_env,
    _get_non_negative_int_env,
    _get_optional_non_negative_int_env,
    _sample_total_cpu_time_s,
    logical_cpu_count,
)


class WorkerBenchmarkObserver:
    """Collect throughput, memory, GPU-clock, and CPU benchmark telemetry."""

    def __init__(
        self,
        *,
        env: Env,
        device: torch.device,
        cpu_pid_list: list[int] | None = None,
    ) -> None:
        self.rank = env.local_rank
        self.device = device
        self.warmup_per_epoch = _get_non_negative_int_env(
            "FROST_THROUGHPUT_WARMUP_BATCHES_PER_EPOCH",
            _get_non_negative_int_env("FROST_THROUGHPUT_WARMUP_BATCHES", 4),
        )
        self.warmup_total = _get_optional_non_negative_int_env(
            "FROST_THROUGHPUT_WARMUP_BATCHES_TOTAL"
        )
        self.measure_target = _get_non_negative_int_env(
            "FROST_THROUGHPUT_MEASURE_BATCHES", 0
        )
        self.measure_start_epoch = _get_non_negative_int_env(
            "FROST_THROUGHPUT_MEASURE_START_EPOCH", 0
        )
        self.gpu_curve_enabled = _get_bool_env("FROST_GPU_UTIL_CURVE_BENCH", False)
        self.cpu_enabled = _get_bool_env("FROST_CPU_UTIL_BENCH", False)

        self.epoch_batches = 0
        self.completed_epochs = 0
        self.total_batches = 0
        self.measured_batches = 0
        self.measured_events = 0
        self.wall_intervals: list[float] = []
        self.gpu_intervals: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.interval_starts: list[float] = []
        self.interval_ends: list[float] = []
        self.reserved_mib: list[float] = []
        self.allocated_mib: list[float] = []

        self._previous_event: torch.cuda.Event | None = None
        self._previous_wall: float | None = None
        self._previous_unix: float | None = None
        self._previous_batch_events: int | None = None
        self._window_start: float | None = None
        self._window_end: float | None = None
        self._last_window_start: float | None = None
        self._last_window_end: float | None = None
        self._last_measured_batches = 0

        self.cpu_pids: list[int] = []
        self.logical_cpus = 0
        self._previous_cpu_total: float | None = None
        self.cpu_used_cores: list[float] = []
        self.cpu_utilization: list[float] = []
        self.cpu_wall_intervals: list[float] = []
        if self.cpu_enabled and self.rank == 0:
            self.cpu_pids = (
                [int(pid) for pid in cpu_pid_list]
                if cpu_pid_list
                else _discover_sibling_rank_pids()
            )
            self.logical_cpus = logical_cpu_count()

    def _past_warmup(self, total_count: int, epoch_count: int) -> bool:
        if self.completed_epochs < self.measure_start_epoch:
            return False
        if self.warmup_total is not None:
            return total_count > self.warmup_total
        return epoch_count > self.warmup_per_epoch

    def on_train_batch_start(self, batch_events: int) -> None:
        """Observe the current start-to-start interval before model execution."""
        self.epoch_batches += 1
        self.total_batches += 1
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream())
        wall = time.perf_counter()
        unix = time.time()
        cpu_total = None
        if self.cpu_enabled and self.rank == 0:
            cpu_total, self.cpu_pids = _sample_total_cpu_time_s(self.cpu_pids)

        can_measure = self.measure_target == 0 or self.measured_batches < self.measure_target
        if (
            can_measure
            and self._previous_event is not None
            and self._previous_wall is not None
            and self._previous_batch_events is not None
            and self._past_warmup(self.total_batches - 1, self.epoch_batches - 1)
        ):
            interval = wall - self._previous_wall
            self.gpu_intervals.append((self._previous_event, event))
            self.wall_intervals.append(interval)
            if self._previous_unix is not None:
                self.interval_starts.append(self._previous_unix)
                self.interval_ends.append(unix)
            self.measured_events += self._previous_batch_events
            self.measured_batches += 1
            if self._window_start is None:
                self._window_start = self._previous_unix
            self._window_end = unix
            if (
                self.cpu_enabled
                and self.rank == 0
                and cpu_total is not None
                and self._previous_cpu_total is not None
                and interval > 0
            ):
                used_cores = max(0.0, cpu_total - self._previous_cpu_total) / interval
                self.cpu_used_cores.append(used_cores)
                self.cpu_utilization.append(used_cores / self.logical_cpus * 100.0)
                self.cpu_wall_intervals.append(interval)

        self._previous_event = event
        self._previous_wall = wall
        self._previous_unix = unix
        self._previous_batch_events = batch_events
        if self.cpu_enabled and self.rank == 0:
            self._previous_cpu_total = cpu_total

    def on_train_batch_end(self) -> None:
        """Sample allocator state after model execution."""
        can_measure = self.measure_target == 0 or len(self.reserved_mib) < self.measure_target
        if can_measure and self._past_warmup(self.total_batches, self.epoch_batches):
            self.reserved_mib.append(torch.cuda.memory_reserved(self.device) / (1024**2))
            self.allocated_mib.append(torch.cuda.memory_allocated(self.device) / (1024**2))

    def on_epoch_end(self) -> None:
        self.completed_epochs += 1
        self.epoch_batches = 0
        self._last_window_start = self._window_start
        self._last_window_end = self._window_end
        self._last_measured_batches = self.measured_batches
        self._previous_event = None
        self._previous_wall = None
        self._previous_unix = None
        self._previous_batch_events = None
        self._previous_cpu_total = None
        self._window_start = None
        self._window_end = None

    @staticmethod
    def _summary(values: list[float], prefix: str) -> dict[str, float]:
        if not values:
            return {}
        ordered = sorted(values)
        count = len(ordered)
        return {
            f"{prefix}_mean": sum(ordered) / count,
            f"{prefix}_p50": ordered[int(0.50 * (count - 1))],
            f"{prefix}_p90": ordered[int(0.90 * (count - 1))],
            f"{prefix}_max": ordered[-1],
        }

    def finalize(self) -> None:
        """Synchronize timing events and emit the established benchmark records."""
        wall_elapsed = sum(self.wall_intervals)
        gpu_elapsed = 0.0
        if self.gpu_intervals:
            torch.cuda.synchronize()
            gpu_elapsed = sum(a.elapsed_time(b) for a, b in self.gpu_intervals) / 1000.0
        wall_rate = self.measured_events / wall_elapsed if wall_elapsed > 0 else 0.0
        gpu_rate = self.measured_events / gpu_elapsed if gpu_elapsed > 0 else 0.0
        if wall_rate > 0:
            print(
                f"TRAIN_THROUGHPUT_PER_GPU rank={self.rank} "
                f"events_per_sec_per_gpu={wall_rate:.1f} "
                f"events_per_sec_per_gpu_gpu_clock={gpu_rate:.1f} "
                f"events={self.measured_events} elapsed_s={wall_elapsed:.3f} "
                f"gpu_elapsed_s={gpu_elapsed:.3f} "
                f"warmup_batches_per_epoch={self.warmup_per_epoch} "
                f"warmup_batches_total={self.warmup_total} "
                f"measure_start_epoch={self.measure_start_epoch} "
                f"measured_batches={self.measured_batches} "
                f"measure_target={self.measure_target}"
            )

        window_start = self._window_start or self._last_window_start
        window_end = self._window_end or self._last_window_end
        window_batches = self.measured_batches or self._last_measured_batches
        if window_start is not None and window_end is not None:
            self._emit(
                "APPENDIX_TRAIN_WINDOW",
                window_start_unix_s=window_start,
                window_end_unix_s=window_end,
                measured_batches=window_batches,
            )
        if self.interval_starts and self.interval_ends:
            self._emit(
                "APPENDIX_TRAIN_INTERVALS",
                measured_batches=len(self.interval_starts),
                interval_start_unix_s=self.interval_starts,
                interval_end_unix_s=self.interval_ends,
            )
        if self.gpu_curve_enabled and self.wall_intervals:
            gpu_ms = [a.elapsed_time(b) for a, b in self.gpu_intervals]
            wall_ms = [seconds * 1000.0 for seconds in self.wall_intervals]
            busy = [g / w * 100.0 if w > 0 else 0.0 for g, w in zip(gpu_ms, wall_ms)]
            self._emit(
                "APPENDIX_TRAIN_GPU_UTIL_CURVE",
                measured_batches=len(busy),
                gpu_ms_curve=[round(value, 4) for value in gpu_ms],
                wall_ms_curve=[round(value, 4) for value in wall_ms],
                gpu_busy_pct_curve=[round(value, 4) for value in busy],
                **self._summary(busy, "gpu_busy_pct"),
            )

        memory = {
            "warmup_batches_per_epoch": self.warmup_per_epoch,
            "warmup_batches_total": self.warmup_total,
            "measure_start_epoch": self.measure_start_epoch,
            "measure_batches_target": self.measure_target,
            "memory_measured_batches": len(self.reserved_mib),
            **self._summary(self.reserved_mib, "reserved_mib"),
            **self._summary(self.allocated_mib, "allocated_mib"),
        }
        self._emit("APPENDIX_TRAIN_MEMORY_SUMMARY", **memory)
        if self.reserved_mib or self.allocated_mib:
            self._emit(
                "APPENDIX_TRAIN_MEMORY_CURVE",
                warmup_batches_per_epoch=self.warmup_per_epoch,
                warmup_batches_total=self.warmup_total,
                measure_start_epoch=self.measure_start_epoch,
                measure_batches_target=self.measure_target,
                measured_batches=len(self.reserved_mib),
                reserved_mib_curve=[round(value, 4) for value in self.reserved_mib],
                allocated_mib_curve=[round(value, 4) for value in self.allocated_mib],
            )
        if self.cpu_enabled and self.rank == 0:
            self._emit(
                "APPENDIX_TRAIN_CPU_UTIL_CURVE",
                warmup_batches_per_epoch=self.warmup_per_epoch,
                warmup_batches_total=self.warmup_total,
                measure_batches_target=self.measure_target,
                logical_cpu_count=self.logical_cpus,
                tracked_pid_count=len(self.cpu_pids),
                tracked_pids=self.cpu_pids,
                measured_batches=len(self.cpu_used_cores),
                used_cores_curve=[round(value, 4) for value in self.cpu_used_cores],
                utilization_pct_curve=[round(value, 4) for value in self.cpu_utilization],
                wall_interval_s_curve=[round(value, 6) for value in self.cpu_wall_intervals],
                **self._summary(self.cpu_used_cores, "used_cores"),
                **self._summary(self.cpu_utilization, "utilization_pct"),
            )

    def _emit(self, marker: str, **payload: object) -> None:
        print(
            f"{marker} "
            f"{json.dumps({'system': 'frost', 'rank': self.rank, **payload}, sort_keys=True)}"
        )
