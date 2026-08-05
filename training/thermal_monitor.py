"""
Jupiter Shot — Independent High-Frequency Thermal Safety Monitor
================================================================
Run 15: Implements an authoritative background-thread thermal monitor that
samples GPU temperature independently of training-step timing.

Design contract:
  - Samples at SAMPLE_INTERVAL_S (default 1.0 s) regardless of step timing.
  - Writes telemetry continuously to <run_dir>/thermal_telemetry.jsonl.
  - Emits a visible [THERMAL WARNING] at >= WARN_C (80°C).
  - At >= STOP_C (90°C), sets a threading.Event that runners poll.
  - If the monitor thread dies unexpectedly, sets a monitor_failed flag.
  - Exactly one active monitor per process (singleton guard via _ACTIVE_MONITOR).
  - Cleanup: stop() joins the thread and clears the singleton.

Usage (in a runner):
    from training.thermal_monitor import ThermalMonitor, THERMAL_WARN_C, THERMAL_STOP_C

    monitor = ThermalMonitor(run_dir=output_dir, run_id=run_id)
    monitor.start()
    try:
        for step in range(1, max_steps + 1):
            if monitor.stop_requested:
                # SAFETY_STOP path
                break
            if monitor.monitor_failed:
                # Non-PASS: monitor health required
                break
            ... training step ...
    finally:
        monitor.stop()
        health = monitor.health_summary()
        # health["healthy"] == True iff monitor ran without failure
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

# ── Thresholds (must not be changed) ─────────────────────────────────────────
THERMAL_WARN_C: int = 80
THERMAL_STOP_C: int = 90
SAMPLE_INTERVAL_S: float = 1.0

# ── Singleton guard ───────────────────────────────────────────────────────────
_ACTIVE_MONITOR: Optional["ThermalMonitor"] = None
_SINGLETON_LOCK = threading.Lock()


def _read_gpu_temp() -> Optional[int]:
    """Read current GPU temperature via nvidia-smi. Returns None on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu",
             "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return int(out) if out.isdigit() else None
    except Exception:
        return None


def _read_gpu_stats() -> dict[str, Any]:
    """
    Read GPU temperature, utilization, VRAM, power, and performance state.
    Returns a dict with all available fields; missing fields are None.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,utilization.gpu,memory.used,"
                "memory.total,power.draw,pstate",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        def _int(v: str) -> Optional[int]:
            try:
                return int(v)
            except ValueError:
                return None
        def _float(v: str) -> Optional[float]:
            try:
                return float(v)
            except ValueError:
                return None
        return {
            "temperature_c": _int(parts[0]) if len(parts) > 0 else None,
            "utilization_pct": _int(parts[1]) if len(parts) > 1 else None,
            "vram_used_mb": _int(parts[2]) if len(parts) > 2 else None,
            "vram_total_mb": _int(parts[3]) if len(parts) > 3 else None,
            "power_draw_w": _float(parts[4]) if len(parts) > 4 else None,
            "pstate": parts[5] if len(parts) > 5 else None,
        }
    except Exception:
        return {
            "temperature_c": None,
            "utilization_pct": None,
            "vram_used_mb": None,
            "vram_total_mb": None,
            "power_draw_w": None,
            "pstate": None,
        }


class ThermalMonitor:
    """
    Independent high-frequency thermal safety monitor.

    Parameters
    ----------
    run_dir : Path
        Directory where thermal_telemetry.jsonl is written.
    run_id : str
        Run identifier embedded in every telemetry record.
    warn_c : int
        Temperature at which a warning is emitted (default 80).
    stop_c : int
        Temperature at which SAFETY_STOP is signalled (default 90).
    interval_s : float
        Sampling interval in seconds (default 1.0).
    temp_source : callable, optional
        Callable returning Optional[int] temperature for testing injection.
        Defaults to _read_gpu_stats.
    """

    def __init__(
        self,
        run_dir: Path,
        run_id: str,
        warn_c: int = THERMAL_WARN_C,
        stop_c: int = THERMAL_STOP_C,
        interval_s: float = SAMPLE_INTERVAL_S,
        temp_source: Any = None,
    ) -> None:
        self._run_dir = Path(run_dir)
        self._run_id = run_id
        self._warn_c = warn_c
        self._stop_c = stop_c
        self._interval_s = interval_s
        self._temp_source = temp_source  # injection point for tests

        # Public state
        self._stop_event = threading.Event()   # set when SAFETY_STOP triggered
        self._shutdown_event = threading.Event()  # set to request graceful stop
        self._failed_event = threading.Event()  # set if monitor thread dies

        self._thread: Optional[threading.Thread] = None
        self._samples: list[dict] = []
        self._telemetry_path: Optional[Path] = None
        self._start_time: Optional[float] = None
        self._stop_time: Optional[float] = None
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def stop_requested(self) -> bool:
        """True when a SAFETY_STOP temperature has been reached."""
        return self._stop_event.is_set()

    @property
    def monitor_failed(self) -> bool:
        """True when the monitor thread died unexpectedly."""
        return self._failed_event.is_set()

    def start(self) -> None:
        """Start the background monitor thread. Raises if already active."""
        global _ACTIVE_MONITOR
        with _SINGLETON_LOCK:
            if _ACTIVE_MONITOR is not None and _ACTIVE_MONITOR is not self:
                raise RuntimeError(
                    "ThermalMonitor: another monitor is already active. "
                    "Call stop() on the existing monitor before starting a new one."
                )
            _ACTIVE_MONITOR = self

        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._telemetry_path = self._run_dir / "thermal_telemetry.jsonl"
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._run,
            name="ThermalMonitor",
            daemon=True,
        )
        self._thread.start()
        print(
            f"[THERMAL MONITOR] Started (warn={self._warn_c}°C, "
            f"stop={self._stop_c}°C, interval={self._interval_s}s). "
            f"Telemetry: {self._telemetry_path}",
            flush=True,
        )

    def stop(self) -> None:
        """Signal the monitor to stop and wait for it to finish."""
        global _ACTIVE_MONITOR
        self._shutdown_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=self._interval_s * 3 + 2)
        self._stop_time = time.time()
        with _SINGLETON_LOCK:
            if _ACTIVE_MONITOR is self:
                _ACTIVE_MONITOR = None
        print("[THERMAL MONITOR] Stopped.", flush=True)

    def health_summary(self) -> dict[str, Any]:
        """
        Return a summary of monitor health and telemetry statistics.
        healthy == True iff the monitor ran without failure and is now stopped.
        """
        with self._lock:
            samples = list(self._samples)
        temps = [s["temperature_c"] for s in samples if s.get("temperature_c") is not None]
        warnings = [s for s in samples if s.get("warning")]
        stop_events = [s for s in samples if s.get("safety_stop")]
        return {
            "healthy": not self._failed_event.is_set() and self._stop_time is not None,
            "monitor_failed": self._failed_event.is_set(),
            "total_samples": len(samples),
            "peak_temperature_c": max(temps) if temps else None,
            "samples_at_or_above_warn_c": len(warnings),
            "samples_at_or_above_stop_c": len(stop_events),
            "safety_stop_triggered": self._stop_event.is_set(),
            "run_id": self._run_id,
            "telemetry_path": str(self._telemetry_path) if self._telemetry_path else None,
            "start_time": self._start_time,
            "stop_time": self._stop_time,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(self) -> None:
        """Background thread body."""
        try:
            while not self._shutdown_event.is_set():
                sample_start = time.time()
                self._sample()
                elapsed = time.time() - sample_start
                sleep_for = max(0.0, self._interval_s - elapsed)
                # Use wait() so shutdown_event wakes us early
                self._shutdown_event.wait(timeout=sleep_for)
        except Exception as exc:
            print(f"[THERMAL MONITOR] Unexpected failure: {exc}", flush=True)
            self._failed_event.set()

    def _sample(self) -> None:
        """Take one telemetry sample and write it to disk."""
        import datetime as _dt
        now_iso = _dt.datetime.now(_dt.timezone.utc).isoformat()

        if self._temp_source is not None:
            # Test injection: callable returns Optional[int] temperature
            temp = self._temp_source()
            stats: dict[str, Any] = {
                "temperature_c": temp,
                "utilization_pct": None,
                "vram_used_mb": None,
                "vram_total_mb": None,
                "power_draw_w": None,
                "pstate": None,
            }
        else:
            stats = _read_gpu_stats()
            temp = stats.get("temperature_c")

        is_warning = temp is not None and temp >= self._warn_c
        is_stop = temp is not None and temp >= self._stop_c

        record: dict[str, Any] = {
            "timestamp": now_iso,
            "run_id": self._run_id,
            **stats,
            "warning": is_warning,
            "safety_stop": is_stop,
        }

        with self._lock:
            self._samples.append(record)

        # Write to telemetry file
        if self._telemetry_path is not None:
            try:
                with open(self._telemetry_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception:
                pass

        # Emit warning
        if is_warning and not is_stop:
            print(
                f"[THERMAL WARNING] GPU {temp}°C >= {self._warn_c}°C "
                f"(threshold {self._stop_c}°C). Training may continue.",
                flush=True,
            )

        # Signal SAFETY_STOP
        if is_stop:
            print(
                f"[THERMAL STOP] GPU {temp}°C >= {self._stop_c}°C. "
                f"Signalling SAFETY_STOP.",
                flush=True,
            )
            self._stop_event.set()
