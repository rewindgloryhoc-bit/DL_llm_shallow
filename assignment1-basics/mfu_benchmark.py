"""Full-step (fwd + bwd) MFU benchmarking via outer-loop CUDA events.

Timing lives only in the training loop so torch.compile stays graph-break free.
"""

from __future__ import annotations

import statistics

import torch

PEAK_TFLOPS_BF16 = 989.0
PEAK_TFLOPS_TF32 = 494.0
WARMUP_STEPS = 15
SUMMARY_WINDOW = 20


def peak_tflops_for_dtype(dtype) -> float:
    if dtype in (torch.bfloat16, torch.float16):
        return PEAK_TFLOPS_BF16
    return PEAK_TFLOPS_TF32


def flops_fwd_layer(B, T, d_model, d_ff) -> float:
    # 2*B*T*(4*d_model^2 + 2*T*d_model) + 6*B*T*d_model*d_ff
    return 2 * B * T * (4 * d_model ** 2 + 2 * T * d_model) + 6 * B * T * d_model * d_ff


def flops_fwd_model(B, T, d_model, d_ff, num_layers) -> float:
    return num_layers * flops_fwd_layer(B, T, d_model, d_ff)


class FullStepMFUBenchmarker:
    """Outer-loop CUDA-event MFU tracker for one training step (fwd+loss / bwd)."""

    def __init__(self, d_model, d_ff, num_layers, dtype):
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.peak_tflops = peak_tflops_for_dtype(dtype)
        self.enabled = torch.cuda.is_available()

        self._start_fwd = None
        self._end_fwd = None
        self._end_bwd = None
        if self.enabled:
            self._start_fwd = torch.cuda.Event(enable_timing=True)
            self._end_fwd = torch.cuda.Event(enable_timing=True)
            self._end_bwd = torch.cuda.Event(enable_timing=True)

        self._step_idx = 0  # counts all steps including warmup
        self._buffer = []  # current 20-step window
        self._all_steps = []  # every post-warmup step (for final report)
        self._window_avgs = []  # each printed window's averages

    def record_start_fwd(self):
        if self.enabled:
            self._start_fwd.record()

    def record_end_fwd(self):
        if self.enabled:
            self._end_fwd.record()

    def record_end_bwd(self):
        if self.enabled:
            self._end_bwd.record()

    @staticmethod
    def _window_average(steps):
        avg = lambda key: statistics.mean(s[key] for s in steps)
        return {
            "n": len(steps),
            "fwd_ms": avg("fwd_ms"),
            "bwd_ms": avg("bwd_ms"),
            "step_ms": avg("step_ms"),
            "fwd_tflops": avg("fwd_tflops"),
            "bwd_tflops": avg("bwd_tflops"),
            "step_tflops": avg("step_tflops"),
            "fwd_mfu": avg("fwd_mfu"),
            "bwd_mfu": avg("bwd_mfu"),
            "step_mfu": avg("step_mfu"),
        }

    def finish_step(self, B, T):
        """Sync once, accumulate metrics (after warmup), maybe print a 20-step window."""
        if not self.enabled:
            return

        torch.cuda.synchronize()
        time_fwd_ms = self._start_fwd.elapsed_time(self._end_fwd)
        time_bwd_ms = self._end_fwd.elapsed_time(self._end_bwd)
        time_step_ms = self._start_fwd.elapsed_time(self._end_bwd)

        self._step_idx += 1
        if self._step_idx <= WARMUP_STEPS:
            return

        flops_fwd = flops_fwd_model(B, T, self.d_model, self.d_ff, self.num_layers)
        flops_bwd = 2.0 * flops_fwd
        flops_step = 3.0 * flops_fwd

        tflops_fwd = flops_fwd / (time_fwd_ms * 1e9) if time_fwd_ms > 0 else 0.0
        tflops_bwd = flops_bwd / (time_bwd_ms * 1e9) if time_bwd_ms > 0 else 0.0
        tflops_step = flops_step / (time_step_ms * 1e9) if time_step_ms > 0 else 0.0

        sample = {
            "fwd_ms": time_fwd_ms,
            "bwd_ms": time_bwd_ms,
            "step_ms": time_step_ms,
            "fwd_tflops": tflops_fwd,
            "bwd_tflops": tflops_bwd,
            "step_tflops": tflops_step,
            "fwd_mfu": (tflops_fwd / self.peak_tflops) * 100.0,
            "bwd_mfu": (tflops_bwd / self.peak_tflops) * 100.0,
            "step_mfu": (tflops_step / self.peak_tflops) * 100.0,
        }
        self._buffer.append(sample)
        self._all_steps.append(sample)

        if len(self._buffer) >= SUMMARY_WINDOW:
            window = self._window_average(self._buffer)
            self._window_avgs.append(window)
            self._print_window_summary(window)
            self._buffer.clear()

    def _print_window_summary(self, window):
        print(
            "===================================================\n"
            f"[Full-Step MFU Summary | {window['n']} steps after warmup "
            f"(peak={self.peak_tflops:.0f} TFLOPS)]\n"
            f"{'phase':<10} {'latency_ms':>12} {'TFLOPS':>10} {'MFU%':>8}\n"
            f"{'Forward':<10} {window['fwd_ms']:>12.3f} {window['fwd_tflops']:>10.2f} {window['fwd_mfu']:>7.2f}%\n"
            f"{'Backward':<10} {window['bwd_ms']:>12.3f} {window['bwd_tflops']:>10.2f} {window['bwd_mfu']:>7.2f}%\n"
            f"{'Full step':<10} {window['step_ms']:>12.3f} {window['step_tflops']:>10.2f} {window['step_mfu']:>7.2f}%\n"
            "==================================================="
        )

    def report_stats(self):
        """Overall + per-window averages for the experiment markdown report."""
        if not self.enabled or not self._all_steps:
            return None
        # Include a partial final window if training stopped mid-buffer.
        windows = list(self._window_avgs)
        if self._buffer:
            windows.append(self._window_average(self._buffer))
        return {
            "peak_tflops": self.peak_tflops,
            "warmup_steps": WARMUP_STEPS,
            "window_size": SUMMARY_WINDOW,
            "valid_steps": len(self._all_steps),
            "overall": self._window_average(self._all_steps),
            "windows": windows,
        }
