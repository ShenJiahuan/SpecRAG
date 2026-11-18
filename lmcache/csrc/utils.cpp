#include <cuda_runtime.h>
#include <linux/perf_event.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <x86intrin.h>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <tuple>
#include "utils.h"

#ifndef RUSAGE_THREAD
#define RUSAGE_THREAD 1
#endif

static long read_cpu_freq_khz(int cpu) {
  if (cpu < 0) return -1;
  char path[128];
  std::snprintf(path, sizeof(path),
                "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu);
  FILE* f = std::fopen(path, "r");
  if (!f) return -1;
  long khz = -1;
  if (std::fscanf(f, "%ld", &khz) != 1) khz = -1;
  std::fclose(f);
  return khz;
}

static double timespec_diff_us(const struct timespec& start,
                               const struct timespec& end) {
  int64_t sec =
      static_cast<int64_t>(end.tv_sec) - static_cast<int64_t>(start.tv_sec);
  int64_t nsec =
      static_cast<int64_t>(end.tv_nsec) - static_cast<int64_t>(start.tv_nsec);
  return static_cast<double>(sec) * 1e6 + static_cast<double>(nsec) / 1e3;
}

static long perf_event_open(struct perf_event_attr* event_attr, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
  return syscall(__NR_perf_event_open, event_attr, pid, cpu, group_fd, flags);
}

static int open_hw_counter(uint64_t config) {
  perf_event_attr event_attr{};
  event_attr.type = PERF_TYPE_HARDWARE;
  event_attr.size = sizeof(event_attr);
  event_attr.config = config;
  event_attr.disabled = 1;
  // Keep these user-space only so the counters describe the measured loop.
  // TSC/thread_cpu can still reveal time lost to interrupts or kernel work.
  event_attr.exclude_kernel = 1;
  event_attr.exclude_hv = 1;
  event_attr.inherit = 0;
  return static_cast<int>(perf_event_open(&event_attr, 0, -1, -1, 0));
}

static long long read_counter(int fd) {
  uint64_t value = 0;
  ssize_t got = read(fd, &value, sizeof(value));
  if (got != static_cast<ssize_t>(sizeof(value))) return -1;
  return static_cast<long long>(value);
}

struct ThreadPerfCounters {
  bool initialized = false;
  bool available = false;
  int cycles_fd = -1;
  int ref_cycles_fd = -1;
  int instructions_fd = -1;

  ~ThreadPerfCounters() {
    if (cycles_fd >= 0) close(cycles_fd);
    if (ref_cycles_fd >= 0) close(ref_cycles_fd);
    if (instructions_fd >= 0) close(instructions_fd);
  }

  void init_once() {
    if (initialized) return;
    initialized = true;

    const char* enable_perf = std::getenv("LMCACHE_CPU_SPIN_PERF");
    if (enable_perf && (enable_perf[0] == '0' || enable_perf[0] == 'n' ||
                        enable_perf[0] == 'N' || enable_perf[0] == 'f' ||
                        enable_perf[0] == 'F')) {
      return;
    }

    cycles_fd = open_hw_counter(PERF_COUNT_HW_CPU_CYCLES);
    ref_cycles_fd = open_hw_counter(PERF_COUNT_HW_REF_CPU_CYCLES);
    instructions_fd = open_hw_counter(PERF_COUNT_HW_INSTRUCTIONS);
    available = cycles_fd >= 0 && ref_cycles_fd >= 0 && instructions_fd >= 0;
    if (!available) {
      if (cycles_fd >= 0) close(cycles_fd);
      if (ref_cycles_fd >= 0) close(ref_cycles_fd);
      if (instructions_fd >= 0) close(instructions_fd);
      cycles_fd = -1;
      ref_cycles_fd = -1;
      instructions_fd = -1;
    }
  }

  bool reset_enable() {
    init_once();
    if (!available) return false;
    ioctl(cycles_fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(ref_cycles_fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(instructions_fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(cycles_fd, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(ref_cycles_fd, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(instructions_fd, PERF_EVENT_IOC_ENABLE, 0);
    return true;
  }

  void disable_read(long long& cycles, long long& ref_cycles,
                    long long& instructions) {
    cycles = -1;
    ref_cycles = -1;
    instructions = -1;
    if (!available) return;
    ioctl(cycles_fd, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(ref_cycles_fd, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(instructions_fd, PERF_EVENT_IOC_DISABLE, 0);
    cycles = read_counter(cycles_fd);
    ref_cycles = read_counter(ref_cycles_fd);
    instructions = read_counter(instructions_fd);
  }
};

std::string get_gpu_pci_bus_id(int device) {
  char pciBusId[13];  // 13 bytes per CUDA doc
  cudaError_t err = cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaDeviceGetPCIBusId failed: ") +
                             cudaGetErrorString(err));
  }
  return std::string(pciBusId);
}

std::tuple<double, double, uint64_t, int, int, long, long, long, long long,
           long long, long long>
cpu_spin_probe(int64_t num_iters) {
  if (num_iters <= 0) {
    return std::make_tuple(0.0, 0.0, (uint64_t)0, -1, -1, (long)-1, (long)-1,
                           (long)-1, (long long)-1, (long long)-1,
                           (long long)-1);
  }
  thread_local ThreadPerfCounters perf_counters;
  perf_counters.init_once();
  rusage usage_before{};
  rusage usage_after{};
  bool have_usage_before = getrusage(RUSAGE_THREAD, &usage_before) == 0;
  int cpu_before = sched_getcpu();
  perf_counters.reset_enable();
  auto start = std::chrono::steady_clock::now();
  struct timespec thread_cpu_start{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &thread_cpu_start);
  uint64_t tsc_start = __rdtsc();
  // volatile sinks prevent the compiler from constant-folding the LCG away.
  volatile uint64_t sink = 0xDEADBEEFCAFEBABEULL;
  uint64_t acc = sink;
  for (int64_t i = 0; i < num_iters; ++i) {
    acc = acc * 6364136223846793005ULL + 1442695040888963407ULL;
    acc ^= (acc >> 17);
  }
  sink = acc;
  (void)sink;
  uint64_t tsc_end = __rdtsc();
  struct timespec thread_cpu_end{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &thread_cpu_end);
  auto end = std::chrono::steady_clock::now();
  long long perf_cycles = -1;
  long long perf_ref_cycles = -1;
  long long perf_instructions = -1;
  perf_counters.disable_read(perf_cycles, perf_ref_cycles, perf_instructions);
  int cpu_after = sched_getcpu();
  bool have_usage_after = getrusage(RUSAGE_THREAD, &usage_after) == 0;
  double wall_us =
      std::chrono::duration<double, std::micro>(end - start).count();
  double thread_cpu_us = timespec_diff_us(thread_cpu_start, thread_cpu_end);
  uint64_t tsc_cycles = tsc_end - tsc_start;
  long voluntary_ctxsw = -1;
  long involuntary_ctxsw = -1;
  if (have_usage_before && have_usage_after) {
    voluntary_ctxsw = usage_after.ru_nvcsw - usage_before.ru_nvcsw;
    involuntary_ctxsw = usage_after.ru_nivcsw - usage_before.ru_nivcsw;
  }
  // Sysfs read happens after timing so it does not perturb the spin.
  long freq_khz = read_cpu_freq_khz(cpu_after);
  return std::make_tuple(wall_us, thread_cpu_us, tsc_cycles, cpu_before,
                         cpu_after, freq_khz, voluntary_ctxsw,
                         involuntary_ctxsw, perf_cycles, perf_ref_cycles,
                         perf_instructions);
}
