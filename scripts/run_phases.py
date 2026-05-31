"""Master orchestration script: Run Phase 1-4 of the MIDI training plan sequentially.

This is designed for unattended overnight execution.
Each phase: extract data -> train with phase-specific config -> verify outputs.
"""
import subprocess
import sys
import os
import time
from pathlib import Path

PYTHON = "E:/conda_envs/stemgen/python"
PROJECT_ROOT = "E:/project/stemgen"

PHASES = [
    {
        "phase": 1,
        "n_tracks": 50,
        "output_dir": "outputs/midi/phase1",
        "epochs_overfit": 0,
        "epochs_train": 50,
        "batch_size": 16,
        "lr": 3e-4,
        "clip_sec": 4.0,
        "skip_overfit": True,
        "use_cache": False,
        "extra_args": ["--d-model", "128", "--num-layers", "2", "--num-heads", "2",
                       "--dropout", "0.3", "--weight-decay", "1e-4"],
        "description": "Pipeline validation - 50 tracks",
        "milestone": "Model generates rhythmic bass MIDI (non-silent)",
    },
    {
        "phase": 2,
        "n_tracks": 200,
        "output_dir": "outputs/midi/phase2",
        "epochs_overfit": 0,
        "epochs_train": 100,
        "batch_size": 16,
        "lr": 3e-4,
        "clip_sec": 4.0,
        "skip_overfit": True,
        "use_cache": False,
        "extra_args": ["--d-model", "128", "--num-layers", "3", "--num-heads", "4",
                       "--dropout", "0.3", "--weight-decay", "1e-4"],
        "description": "Baseline results - 200 tracks",
        "milestone": "activity_accuracy > 70%, pitch_accuracy > 50%",
    },
    {
        "phase": 3,
        "n_tracks": 550,
        "output_dir": "outputs/midi/phase3",
        "epochs_overfit": 0,
        "epochs_train": 200,
        "batch_size": 16,
        "lr": 3e-4,
        "clip_sec": 4.0,
        "skip_overfit": True,
        "use_cache": True,
        "extra_args": ["--d-model", "192", "--num-layers", "4", "--num-heads", "4",
                       "--dropout", "0.3", "--weight-decay", "1e-4"],
        "description": "Audio E2 alignment - 550 tracks",
        "milestone": "activity F1 > 0.80",
    },
    {
        "phase": 4,
        "n_tracks": 1000,
        "output_dir": "outputs/midi/phase4",
        "epochs_overfit": 0,
        "epochs_train": 200,
        "batch_size": 16,
        "lr": 3e-4,
        "clip_sec": 4.0,
        "skip_overfit": True,
        "use_cache": True,
        "extra_args": ["--d-model", "192", "--num-layers", "4", "--num-heads", "4",
                       "--dropout", "0.3", "--weight-decay", "1e-4"],
        "description": "Extended validation - 1000 tracks",
        "milestone": "Verify more data improves results",
    },
]

LOG_FILE = os.path.join(PROJECT_ROOT, "outputs/midi", "phase_run.log")


def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def run_cmd(cmd, description, timeout=None):
    """Run a command and log output."""
    log(f"Running: {description}")
    log(f"Command: {' '.join(cmd)}")

    start = time.time()
    try:
        result = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
        elapsed = time.time() - start

        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                log(f"  {line}")

        if result.returncode != 0:
            log(f"ERROR: {description} failed (rc={result.returncode})")
            if result.stderr:
                log(f"STDERR: {result.stderr[:2000]}")
            return False

        log(f"Completed: {description} ({elapsed/60:.1f} min)")
        return True
    except subprocess.TimeoutExpired:
        log(f"TIMEOUT: {description} exceeded {timeout}s")
        return False
    except Exception as e:
        log(f"EXCEPTION: {description}: {e}")
        return False


def verify_outputs(phase_num, output_dir):
    """Check that training produced valid outputs."""
    odir = os.path.join(PROJECT_ROOT, output_dir)

    checks = {
        "pianoroll_comparison.png": os.path.join(odir, "figures", "pianoroll_comparison.png"),
        "loss_curves.png": os.path.join(odir, "figures", "loss_curves.png"),
        "generated_bass.mid": os.path.join(odir, "generated_bass.mid"),
        "best checkpoint": os.path.join(odir, "checkpoints", "best.pt"),
    }

    all_ok = True
    for name, path in checks.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            if name.endswith(".mid") and size < 100:
                log(f"  WARNING: {name} too small ({size} bytes)")
                all_ok = False
            else:
                log(f"  OK: {name} ({size/1024:.1f}KB)" if size > 1024 else f"  OK: {name} ({size} bytes)")
        else:
            log(f"  MISSING: {name}")
            all_ok = False

    return all_ok


def main():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    log("=" * 60)
    log("MIDI Training Pipeline: Phases 1-4")
    log(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    results = []

    for i, config in enumerate(PHASES):
        phase_num = config["phase"]
        log(f"\n{'=' * 60}")
        log(f"PHASE {phase_num}: {config['description']}")
        log(f"  Tracks: {config['n_tracks']}, Epochs: {config['epochs_train']}")
        log(f"  Milestone: {config['milestone']}")
        log(f"{'=' * 60}")

        # Check disk space before extracting
        import shutil
        _, _, free = shutil.disk_usage("E:/")
        log(f"  Free disk space: {free/1e9:.1f}GB")

        # Step 1: Extract data (incremental - skips already extracted tracks)
        extract_cmd = [
            PYTHON, "scripts/extract_midi_dataset.py",
            "--n-tracks", str(config["n_tracks"]),
            "--out-dir", "dataset/midi_subset",
        ]
        extract_timeout = 7200  # 2 hours for extraction (scanning 50k archive entries is slow)
        if not run_cmd(extract_cmd, f"Phase {phase_num}: Extract {config['n_tracks']} tracks", timeout=extract_timeout):
            log(f"Phase {phase_num} extraction failed, continuing to next phase")
            results.append({"phase": phase_num, "result": "FAILED (extraction)"})
            continue

        # Step 2: Train
        train_cmd = [
            PYTHON, "scripts/train_midi.py",
            "--max-tracks", str(config["n_tracks"]),
            "--output-dir", config["output_dir"],
            "--epochs-overfit", str(config["epochs_overfit"]),
            "--epochs-train", str(config["epochs_train"]),
            "--batch-size", str(config["batch_size"]),
            "--lr", str(config["lr"]),
            "--clip-sec", str(config["clip_sec"]),
        ]
        if config["skip_overfit"]:
            train_cmd.append("--skip-overfit")
        if config["use_cache"]:
            train_cmd.append("--use-cache")
        if config.get("extra_args"):
            train_cmd.extend(config["extra_args"])

        # Generous timeout for overnight unattended runs (12 hours per phase)
        timeout_seconds = 43200  # 12 hours
        log(f"  Training timeout: {timeout_seconds/3600:.1f} hours")

        if not run_cmd(train_cmd, f"Phase {phase_num}: Train on {config['n_tracks']} tracks", timeout=timeout_seconds):
            log(f"Phase {phase_num} training failed")
            results.append({"phase": phase_num, "result": "FAILED (training)"})
            continue

        # Step 3: Verify
        log(f"Verifying Phase {phase_num} outputs...")
        if verify_outputs(phase_num, config["output_dir"]):
            log(f"Phase {phase_num} COMPLETE - all checks passed")
            results.append({"phase": phase_num, "result": "SUCCESS"})
        else:
            log(f"Phase {phase_num} complete but some checks failed")
            results.append({"phase": phase_num, "result": "PARTIAL"})

    # Final summary
    log(f"\n{'=' * 60}")
    log("ALL PHASES COMPLETE")
    log(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'=' * 60}")
    for r in results:
        log(f"  Phase {r['phase']}: {r['result']}")
    log("=" * 60)


if __name__ == "__main__":
    main()
