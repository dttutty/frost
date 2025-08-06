# In a utility file like my_timer.py or utils.py

import time


class Profiler:
    """A simple profiler to accumulate and report timing statistics."""
    def __init__(self):
        self.times = {
            '0_compute_uniq_node_id': 0.0,
            '1_transfer_node_feat': 0.0,
            '2_transfer_node_mapping': 0.0,
            
            '0_compute_uniq_edge_id': 0.0,
            '1_transfer_edge_feat': 0.0,
            '2_transfer_edge_mapping': 0.0,
            
            '3_transfer_node_mem': 0.0,
            '4_transfer_node_mail': 0.0,
            
            '5_mapping_back': 0.0,
            
            '6_compute_uniq_root_node_id_for_mem_update': 0.0,
            '7_transfer_back_root_node_mem': 0.0
        }
        
        # Initialize all required count metrics to zero
        self.counts = {
            'batch_count': 0,
            'unique_edge_id_count': 0,
            'total_edge_id_count': 0,
            'total_node_id_count': 0,
            'unique_node_id_count': 0,
            'unique_root_node_count': 0,
            'total_root_node_count': 0
        }
        
    
    def next_batch(self):
        """Increment the batch counter."""
        self.counts['batch_count'] += 1


    def increase_time(self, time_stats: dict):
        """Accumulate timing stats from a dictionary."""
        for key, value in time_stats.items():
            if key in self.times:
                self.times[key] += value
            else:
                print(f"Warning: Timing metric '{key}' not defined in Profiler.")

                
    def increase_count(self, count_stats: dict):
        """Accumulate count stats from a dictionary."""
        for key, value in count_stats.items():
            if key in self.counts:
                self.counts[key] += value
            else:
                print(f"Warning: Count metric '{key}' not defined in Profiler.")
                
    def report(self):
        """Print a formatted final report of all collected statistics."""
        batch_count = self.counts['batch_count']
        if batch_count == 0:
            print("Profiler Report: No data collected (0 batches).")
            return

        print("\n" + "="*80)
        print(" " * 28 + "PROFILING FINAL REPORT")
        print("="*80)
        print(f"Total Batches Processed: {batch_count}\n")
        
        # --- Timing Statistics Report ---
        print("--- Timing Statistics ---")
        print(f"{'Metric':<50} | {'Total (s)':>12} | {'Avg/Batch (ms)':>15}")
        print("-" * 80)
        # Sort items by key to ensure consistent order
        for name, total_time in sorted(self.times.items()):
            avg_time_ms = (total_time / batch_count) * 1000
            print(f"{name:<50} | {total_time:>12.4f} | {avg_time_ms:>15.4f}")
        
        print("\n" + "--- Count Statistics ---")
        print(f"{'Metric':<50} | {'Total Count':>12} | {'Avg/Batch':>15}")
        print("-" * 80)
        # Sort items by key to ensure consistent order
        for name, total_count in sorted(self.counts.items()):
            if name == 'batch_count': continue # Don't report batch_count in the table
            avg_count = total_count / batch_count
            print(f"{name:<50} | {total_count:>12d} | {avg_count:>15.2f}")
            
        print("="*80)
        
        self.reset()
        
    def reset(self):
        """Reset all timing and count statistics."""
        for key in self.times:
            self.times[key] = 0.0
        for key in self.counts:
            self.counts[key] = 0

# Also include your Timer class here if it's in the same file
class Timer:
    """A simple timer context manager that stores the duration of the last run."""
    def __init__(self, name='Task'):
        self.name = name
        self.last_duration = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        self.last_duration = end_time - self.start_time
        
profiler = Profiler()
