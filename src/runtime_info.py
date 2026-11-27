"""Measure process-wide peak memory on Windows and Unix hosts."""

import json
import os
import sys


def save_peak_memory(output):
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("size", wintypes.DWORD), ("page_faults", wintypes.DWORD),
                    ("peak_working_set", ctypes.c_size_t), ("working_set", ctypes.c_size_t),
                    ("peak_paged_pool", ctypes.c_size_t), ("paged_pool", ctypes.c_size_t),
                    ("peak_nonpaged_pool", ctypes.c_size_t), ("nonpaged_pool", ctypes.c_size_t),
                    ("pagefile_usage", ctypes.c_size_t), ("peak_pagefile_usage", ctypes.c_size_t),
                ]

            process = ctypes.windll.kernel32.GetCurrentProcess
            process.restype = wintypes.HANDLE
            query = ctypes.windll.psapi.GetProcessMemoryInfo
            query.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
            counters = ProcessMemoryCounters()
            counters.size = ctypes.sizeof(counters)
            peak = counters.peak_working_set if query(process(), ctypes.byref(counters), counters.size) else None
        else:
            import resource

            maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak = maxrss if sys.platform == "darwin" else 1024 * maxrss
    except (ImportError, OSError, AttributeError):
        peak = None

    (output / "process_runtime.json").write_text(json.dumps({
        "peak_resident_bytes": peak,
        "scope": "Whole experiment process through validation and artifact export",
        "thread_environment": {key: os.environ.get(key) for key in
                               ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]},
    }, indent=2), encoding="utf-8")
