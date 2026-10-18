import re
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt

'''
Writes a per-experiment markdown report plus loss curve:
    experiments/<experiment_id>/report.md
    experiments/<experiment_id>/loss_curve.png
'''

REPORT_DIR = Path(__file__).resolve().parent / "experiments"
TIME_FMT = "%Y-%m-%d %H:%M:%S"


def default_experiment_id():
    return time.strftime("run_%Y%m%d_%H%M%S", time.localtime())


def sanitize_experiment_id(experiment_id):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", experiment_id).strip("_") or default_experiment_id()


def _format_duration(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(hours)}h {int(minutes)}m {secs:.1f}s ({seconds:.1f} s)"


def save_loss_curve(steps, loss_set, path, title):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, loss_set, linewidth=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _append_mfu_section(lines, mfu_stats):
    if not mfu_stats:
        return
    overall = mfu_stats["overall"]
    lines += [
        "",
        "## Full-step MFU",
        "",
        f"- **Peak TFLOPS (denominator):** {mfu_stats['peak_tflops']:.0f}",
        f"- **Warmup steps discarded:** {mfu_stats['warmup_steps']}",
        f"- **Valid post-warmup steps:** {mfu_stats['valid_steps']}",
        f"- **Terminal print window:** {mfu_stats['window_size']} steps",
        "",
        "### Overall average (all valid steps)",
        "",
        "| phase | latency_ms | TFLOPS | MFU% |",
        "| --- | --- | --- | --- |",
        f"| Forward | {overall['fwd_ms']:.3f} | {overall['fwd_tflops']:.2f} | {overall['fwd_mfu']:.2f}% |",
        f"| Backward | {overall['bwd_ms']:.3f} | {overall['bwd_tflops']:.2f} | {overall['bwd_mfu']:.2f}% |",
        f"| Full step | {overall['step_ms']:.3f} | {overall['step_tflops']:.2f} | {overall['step_mfu']:.2f}% |",
    ]
    windows = mfu_stats.get("windows") or []
    if windows:
        lines += [
            "",
            "### Per-window averages",
            "",
            "| window | n | fwd_ms | bwd_ms | step_ms | fwd_MFU% | bwd_MFU% | step_MFU% |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for i, w in enumerate(windows, start=1):
            lines.append(
                f"| {i} | {w['n']} | {w['fwd_ms']:.3f} | {w['bwd_ms']:.3f} | {w['step_ms']:.3f} | "
                f"{w['fwd_mfu']:.2f}% | {w['bwd_mfu']:.2f}% | {w['step_mfu']:.2f}% |"
            )


def write_report(experiment_id, start_time, end_time, start_step, step_times, loss_set, hyperparams,
                 interrupted=False, notes=None, mfu_stats=None, report_dir=REPORT_DIR):
    out_dir = Path(report_dir) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "loss_curve.png"
    report_path = out_dir / "report.md"

    steps = list(range(start_step, start_step + len(loss_set)))
    if loss_set:
        save_loss_curve(steps, loss_set, plot_path, f"{experiment_id}: training loss")

    lines = [
        f"# Experiment: {experiment_id}",
        "",
    ]
    if notes:
        lines += ["## Brief summary", "", notes, ""]
    lines += [
        "## Timing",
        "",
        f"- **Training start time:** {time.strftime(TIME_FMT, time.localtime(start_time))}",
        f"- **Training end time:** {time.strftime(TIME_FMT, time.localtime(end_time))}",
        f"- **Total training time:** {_format_duration(end_time - start_time)}",
        f"- **Steps completed:** {len(step_times)} (step {start_step} to {start_step + len(step_times) - 1})",
    ]
    if interrupted:
        lines.append("- **Status:** interrupted (Ctrl-C) before reaching the final step")

    if step_times:
        # First step includes torch.compile warm-up, so it is reported separately.
        steady = step_times[1:] or step_times
        lines += [
            "",
            "### Per-step time",
            "",
            "| statistic | seconds |",
            "| --- | --- |",
            f"| mean (excluding first step) | {statistics.mean(steady):.4f} |",
            f"| median | {statistics.median(steady):.4f} |",
            f"| min | {min(steady):.4f} |",
            f"| max | {max(steady):.4f} |",
            f"| first step (includes compile) | {step_times[0]:.4f} |",
        ]

    lines += [
        "",
        "## Hyperparameters",
        "",
        "| name | value |",
        "| --- | --- |",
    ]
    lines += [f"| {name} | {value} |" for name, value in hyperparams.items()]

    _append_mfu_section(lines, mfu_stats)

    if loss_set:
        tail = loss_set[-100:]
        lines += [
            "",
            "## Loss",
            "",
            f"- **Initial loss:** {loss_set[0]:.4f}",
            f"- **Final loss:** {loss_set[-1]:.4f}",
            f"- **Mean of last {len(tail)} steps:** {statistics.mean(tail):.4f}",
            f"- **Min loss:** {min(loss_set):.4f}",
            "",
            f"![loss curve]({plot_path.name})",
        ]

    report_path.write_text("\n".join(lines) + "\n")
    return report_path
