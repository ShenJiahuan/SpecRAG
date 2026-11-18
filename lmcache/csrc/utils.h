#include <cstdint>
#include <string>
#include <tuple>

std::string get_gpu_pci_bus_id(int device);

// Returns (wall_us, thread_cpu_us, tsc_cycles, cpu_before, cpu_after, freq_khz,
// voluntary_ctxsw, involuntary_ctxsw, perf_cycles, perf_ref_cycles,
// perf_instructions). The spin is a fixed-count register-only arithmetic loop.
// wall_us measures elapsed time; thread_cpu_us measures CPU time charged to
// this thread; tsc_cycles is from rdtsc (invariant on modern x86, so
// wall-time-proportional). cpu_before/after detect core migration; freq_khz is
// scaling_cur_freq read after the loop; ctxsw counts come from
// getrusage(RUSAGE_THREAD) before/after the loop. perf_* are user-space
// per-thread hardware counters, or -1 when perf_event_open is unavailable.
std::tuple<double, double, uint64_t, int, int, long, long, long, long long,
           long long, long long>
cpu_spin_probe(int64_t num_iters);
