"""Native process RSS sampling and synchronized FP32 inference timings."""
from __future__ import annotations

import ctypes
import os
import threading
import time
import numpy as np
import torch


def process_rss_bytes() -> int:
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if not query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.WorkingSetSize)
    # /proc is current RSS, unlike resource.ru_maxrss (process lifetime peak).
    with open("/proc/self/statm", encoding="ascii") as stream:
        return int(stream.read().split()[1]) * int(os.sysconf("SC_PAGE_SIZE"))


class RSSMonitor:
    """Sample process RSS; includes native tensors, excludes other processes."""
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.peak = process_rss_bytes()
        self.start_bytes = self.peak
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self) -> None:
        while not self.stop_event.wait(.01):
            self.peak = max(self.peak, process_rss_bytes())

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> int:
        self.peak = max(self.peak, process_rss_bytes())
        self.stop_event.set()
        self.thread.join()
        return self.peak


def benchmark_inference(model, features, device, *, repeats=30, warmup=5):
    """Time the actual resident model; exclude host transfer and wire decode."""
    if repeats < 2 or warmup < 0 or len(features) == 0:
        raise ValueError("Invalid inference benchmark settings")
    modes = {module: module.training for module in model.modules()}
    model.eval()
    x = features.to(device)
    timings = []
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    try:
        with torch.inference_mode():
            for _ in range(warmup):
                model(x)
            sync()
            for _ in range(repeats):
                sync()
                start = time.perf_counter()
                model(x)
                sync()
                timings.append((time.perf_counter() - start) * 1000)
    finally:
        for module, training in modes.items():
            module.training = training
    return {
        "execution_dtype": "fp32", "integer_kernel": False,
        "device": str(device), "batch_size": len(x), "repeats": repeats,
        "warmup": warmup, "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "median_ms_per_example": float(np.median(timings) / len(x)),
        "scope": "resident_model_forward_only_excludes_wire_decode_and_transfer",
    }
