import os
import argparse
import threading
import queue
import time
from collections import defaultdict
import torch

# Optional torch.distributed support
try:
    import torch.distributed as dist
except Exception:
    dist = None


# ---------------- Simulated ops ----------------
def mfgs_to_cuda(batch):
    print(f"  (Feeder)  Step 1: mfgs_to_cuda for batch {batch['id']}...")
    time.sleep(0.2)
    batch["nfeat_src"] = f"nfeat_for_{batch['id']}"
    batch["efeat_src"] = f"efeat_for_{batch['id']}"
    batch["mail_src"] = f"mail_for_{batch['id']}"
    return batch


def copy_to_gpu(data_src, type):
    delay = {"nfeat": 0.8, "efeat": 0.5, "mail": 0.3}[type]
    print(
        f"    ({type.upper()}_Copy)  Step 2: Copying '{data_src}' (takes {delay}s)..."
    )
    time.sleep(delay)
    return f"{data_src}_on_gpu"


def compute_on_gpu(gpu_batch):
    print(
        f"        (MAIN_THREAD::Compute)  Computing on assembled batch {gpu_batch['id']}..."
    )
    time.sleep(1.0)
    return {"id": gpu_batch["id"], "result": f"result_of_{gpu_batch['id']}_on_gpu"}


def move_to_cpu(gpu_result):
    print(f"          (MoverCPU) Moving result for batch {gpu_result['id']} to CPU...")
    time.sleep(0.1)
    return {
        "id": gpu_result["id"],
        "result": gpu_result["result"].replace("_on_gpu", "_on_cpu"),
    }


class AsyncStallCoordinator:
    """
    Barrier-free coordinator:
    - Tracks local contiguous CPU-done prefix (from 0 upward).
    - A background sync thread periodically all-reduces MIN(local_prefix_done)
      to compute global_prefix_done across ranks.
    - mail worker waits until global_prefix_done >= (batch_id - mem_stallness).
    """

    def __init__(
        self,
        mem_stallness: int,
        rank: int = 0,
        world_size: int = 1,
        use_dist: bool = False,
        sync_interval_ms: int = 10,
    ):
        self.mem_stallness = int(mem_stallness)
        self.rank = rank
        self.world_size = world_size
        self.use_dist = bool(use_dist and dist is not None and dist.is_initialized())
        self.sync_interval_ms = int(sync_interval_ms)

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

        # Local tracking
        self._local_done = set()
        self._local_prefix_done = -1  # largest t s.t. [0..t] are done locally
        # Global (updated by sync thread)
        self._global_prefix_done = -1

        self._stop = False
        self._sync_thread = threading.Thread(
            target=self._sync_loop, name=f"AsyncStallSync_r{rank}", daemon=True
        )

    def start(self):
        self._sync_thread.start()

    def stop(self):
        self._stop = True
        # Wake waiters so they can re-check condition and exit
        with self._cond:
            self._cond.notify_all()
        # Best-effort join
        self._sync_thread.join(timeout=2.0)

    def _recompute_local_prefix(self):
        # Advance local contiguous prefix as far as possible
        while (self._local_prefix_done + 1) in self._local_done:
            self._local_prefix_done += 1
            # keep set size bounded
            self._local_done.discard(self._local_prefix_done)

    def on_cpu_done(self, batch_id: int):
        # Called by CPU mover after finishing a batch locally
        with self._lock:
            self._local_done.add(int(batch_id))
            self._recompute_local_prefix()
        # No barrier here; background thread will publish global state

    def wait_mail_permission(self, batch_id: int):
        if self.mem_stallness <= 0:
            return
        threshold = int(batch_id) - self.mem_stallness
        if threshold < 0:
            return
        with self._cond:
            while self._global_prefix_done < threshold and not self._stop:
                self._cond.wait(timeout=0.02)  # short timed waits to avoid dead waits

    def _sync_loop(self):
        # All ranks call all_reduce at a steady cadence to keep call counts aligned.
        # This thread is independent; it won't block the main/data-prep threads.
        while not self._stop:
            local_val = -1
            with self._lock:
                local_val = self._local_prefix_done
            if self.use_dist and torch is not None:
                try:
                    t = torch.tensor([local_val], dtype=torch.int64)
                    dist.all_reduce(t, op=dist.ReduceOp.MIN)
                    global_val = int(t.item())
                except Exception:
                    global_val = local_val
            else:
                global_val = local_val

            updated = False
            with self._cond:  # same lock as waiters
                if global_val > self._global_prefix_done:
                    self._global_prefix_done = global_val
                    updated = True
                if updated:
                    self._cond.notify_all()

            time.sleep(self.sync_interval_ms / 1000.0)

        # Final notify to release any waiters on shutdown
        with self._cond:
            self._cond.notify_all()

    # Optional helpers for debugging
    @property
    def global_prefix_done(self):
        with self._lock:
            return self._global_prefix_done

    @property
    def local_prefix_done(self):
        with self._lock:
            return self._local_prefix_done


# ---------------- Thread classes ----------------
class MFGFeederThread(threading.Thread):
    def __init__(self, source_q, nfeat_q, efeat_q, mail_q):
        super().__init__(name="Feeder_Thread")
        self.source_q, self.nfeat_q, self.efeat_q, self.mail_q = (
            source_q,
            nfeat_q,
            efeat_q,
            mail_q,
        )

    def run(self):
        while True:
            raw_batch = self.source_q.get()
            if raw_batch is None:
                self.nfeat_q.put(None)
                self.efeat_q.put(None)
                self.mail_q.put(None)
                break
            processed_batch = mfgs_to_cuda(raw_batch)
            batch_id = processed_batch["id"]
            self.nfeat_q.put({"id": batch_id, "data": processed_batch["nfeat_src"]})
            self.efeat_q.put({"id": batch_id, "data": processed_batch["efeat_src"]})
            self.mail_q.put({"id": batch_id, "data": processed_batch["mail_src"]})


class GPUCopyWorkerThread(threading.Thread):
    def __init__(self, task_type, source_q, results_q, pre_wait=None):
        super().__init__(name=f"{task_type.upper()}_Worker")
        self.task_type, self.source_q, self.results_q = task_type, source_q, results_q
        self.pre_wait = pre_wait  # optional callable(batch_id) to wait before copying

    def run(self):
        while True:
            task = self.source_q.get()
            if task is None:
                self.results_q.put(None)
                break
            batch_id = task["id"]
            if self.pre_wait is not None:
                self.pre_wait(batch_id)
            gpu_data = copy_to_gpu(task["data"], self.task_type)
            self.results_q.put(
                {"id": batch_id, "type": self.task_type, "data": gpu_data}
            )


class AggregatorThread(threading.Thread):
    def __init__(self, source_q, dest_q, num_workers):
        super().__init__(name="Aggregator_Thread")
        self.source_q, self.dest_q, self.num_workers = source_q, dest_q, num_workers
        self.pending_batches = defaultdict(dict)

    def run(self):
        finished_workers = 0
        while finished_workers < self.num_workers:
            partial_result = self.source_q.get()
            if partial_result is None:
                finished_workers += 1
                continue
            batch_id, part_type, part_data = (
                partial_result["id"],
                partial_result["type"],
                partial_result["data"],
            )
            self.pending_batches[batch_id][part_type] = part_data
            if len(self.pending_batches[batch_id]) == self.num_workers:
                print(f"      (Aggregator) Assembled complete batch {batch_id}!")
                complete_batch = {"id": batch_id, **self.pending_batches.pop(batch_id)}
                self.dest_q.put(complete_batch)
        self.dest_q.put(None)


class DataMoveToCPUThread(threading.Thread):
    def __init__(self, source_q, coordinator: AsyncStallCoordinator):
        super().__init__(name="CPU_Mover_Thread")
        self.source_q = source_q
        self.coordinator = coordinator

    def run(self):
        while True:
            gpu_result = self.source_q.get()
            if gpu_result is None:
                break
            cpu_result = move_to_cpu(gpu_result)
            print(
                f"            (Final) Result for batch {cpu_result['id']} is ready on CPU: {cpu_result['result']}"
            )
            # Inform coordinator (with optional distributed barrier per batch)
            self.coordinator.on_cpu_done(cpu_result["id"])
        print("CPU Mover Thread Finished.")


# ---------------- Main ----------------
def maybe_init_dist(backend_env: str):
    if dist is None:
        return False, 0, 1
    # If launched via torchrun, env vars exist
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        backend = os.environ.get("DIST_BACKEND", backend_env or "nccl")
        # Fallback to gloo if nccl not usable in this environment
        try:
            dist.init_process_group(backend=backend, init_method="env://")
        except Exception:
            dist.init_process_group(backend="gloo", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"[Dist] Initialized: rank={rank}, world_size={world_size}")
        return True, rank, world_size
    return False, rank, world_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_batches", type=int, default=25)
    parser.add_argument("--mem_stallness", type=int, default=6)
    parser.add_argument("--backend", type=str, default="nccl", help="nccl or gloo")
    args = parser.parse_args()

    use_dist, rank, world_size = maybe_init_dist(args.backend)
    world_size = world_size or 1  # safety

    # Queues
    raw_data_queue = queue.Queue(maxsize=1)
    nfeat_q, efeat_q, mail_q = (
        queue.Queue(maxsize=1),
        queue.Queue(maxsize=1),
        queue.Queue(maxsize=1),
    )
    partial_results_queue = queue.Queue(maxsize=1)
    compute_queue = queue.Queue(maxsize=1)
    cpu_bound_queue = queue.Queue()

    # Coordinator
    coordinator = AsyncStallCoordinator(
        mem_stallness=args.mem_stallness,
        rank=rank,
        world_size=world_size,
        use_dist=use_dist,
    )
    coordinator.start()

    # Threads
    feeder = MFGFeederThread(raw_data_queue, nfeat_q, efeat_q, mail_q)
    nfeat_worker = GPUCopyWorkerThread("nfeat", nfeat_q, partial_results_queue)
    efeat_worker = GPUCopyWorkerThread("efeat", efeat_q, partial_results_queue)
    mail_worker = GPUCopyWorkerThread(
        "mail", mail_q, partial_results_queue, pre_wait=coordinator.wait_mail_permission
    )
    aggregator = AggregatorThread(partial_results_queue, compute_queue, num_workers=3)
    mover_to_cpu = DataMoveToCPUThread(cpu_bound_queue, coordinator)

    background_threads = [
        feeder,
        nfeat_worker,
        efeat_worker,
        mail_worker,
        aggregator,
        mover_to_cpu,
    ]
    for t in background_threads:
        t.start()

    # Enqueue only batches assigned to this rank (round-robin by modulo)
    num_batches = args.num_batches
    for i in range(num_batches):
        if (
            i % world_size
        ) == rank:  # e.g., rank0: 0,3,6; rank1: 1,4,7; rank2: 2,5,8; ...
            raw_data_queue.put({"id": i})
    raw_data_queue.put(None)

    print("\n" + "=" * 25 + f" MAIN THREAD COMPUTE LOOP (rank {rank}) " + "=" * 25)
    processed_batches = 0
    expected_batches = sum(1 for i in range(num_batches) if (i % world_size) == rank)

    while processed_batches < expected_batches:
        gpu_data_batch = compute_queue.get()
        if gpu_data_batch is None:
            break
        # Extra safety: skip if not my shard (shouldn't happen because feeder already sharded)
        if (gpu_data_batch.get("id", -1) % world_size) != rank:
            continue
        compute_result = compute_on_gpu(gpu_data_batch)
        cpu_bound_queue.put(compute_result)
        processed_batches += 1

    print("=" * 24 + " MAIN THREAD LOOP FINISHED " + "=" * 25 + "\n")

    # Shutdown CPU mover
    cpu_bound_queue.put(None)

    print("MAIN THREAD: Waiting for all background threads to join...")
    for t in background_threads:
        t.join()

    # Optional: clean up dist
    if use_dist and dist is not None and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    coordinator.stop()
    print("Program finished successfully.")


if __name__ == "__main__":
    main()
