"""
Scaling experiments: HuBERT data scaling, CRNN architecture scaling, combined.
Runs multiple experiments sequentially and collects results.
"""
import subprocess, os, time, json, sys

PY = "E:/conda_envs/stemgen/python"
ROOT = "E:/project/stemgen"
DATA = "dataset/midi_subset"

EXPERIMENTS = [
    # === HuBERT data scaling ===
    {
        "name": "HuBERT-GRU-550",
        "script": "scripts/train_hubert.py",
        "desc": "HuBERT-base + GRU @ 550 tracks",
        "args": ["--max-tracks", "550", "--output-dir", "outputs/midi/scale_hubert_gru_550",
                 "--epochs-train", "30", "--batch-size", "8", "--lr", "1e-3",
                 "--head-type", "gru", "--dropout", "0.3", "--weight-decay", "1e-4",
                 "--early-stop-patience", "15"],
    },
    # === HuBERT + CRNN (new combo) ===
    {
        "name": "HuBERT-CRNN-200",
        "script": "scripts/train_hubert.py",
        "desc": "HuBERT-base + CRNN @ 200 tracks",
        "args": ["--max-tracks", "200", "--output-dir", "outputs/midi/scale_hubert_crnn_200",
                 "--epochs-train", "30", "--batch-size", "8", "--lr", "1e-3",
                 "--head-type", "transformer", "--dropout", "0.3", "--weight-decay", "1e-4",
                 "--early-stop-patience", "15"],
    },
    {
        "name": "HuBERT-CRNN-550",
        "script": "scripts/train_hubert.py",
        "desc": "HuBERT-base + CRNN @ 550 tracks",
        "args": ["--max-tracks", "550", "--output-dir", "outputs/midi/scale_hubert_crnn_550",
                 "--epochs-train", "30", "--batch-size", "8", "--lr", "1e-3",
                 "--head-type", "transformer", "--dropout", "0.3", "--weight-decay", "1e-4",
                 "--early-stop-patience", "15"],
    },
    # === CRNN standalone scaling (larger model) ===
    {
        "name": "CRNN-L-200",
        "script": "scripts/train_crnn.py",
        "desc": "CRNN-large (hidden=512) @ 200 tracks",
        "args": ["--max-tracks", "200", "--output-dir", "outputs/midi/scale_crnn_l_200",
                 "--epochs-train", "30", "--batch-size", "16", "--lr", "1e-3",
                 "--dropout", "0.3", "--weight-decay", "1e-4",
                 "--early-stop-patience", "15", "--use-mix",
                 "--hidden", "512"],
    },
    {
        "name": "CRNN-L-550",
        "script": "scripts/train_crnn.py",
        "desc": "CRNN-large (hidden=512) @ 550 tracks",
        "args": ["--max-tracks", "550", "--output-dir", "outputs/midi/scale_crnn_l_550",
                 "--epochs-train", "30", "--batch-size", "16", "--lr", "1e-3",
                 "--dropout", "0.3", "--weight-decay", "1e-4",
                 "--early-stop-patience", "15", "--use-mix",
                 "--hidden", "512"],
    },
    {
        "name": "CRNN-L-1000",
        "script": "scripts/train_crnn.py",
        "desc": "CRNN-large (hidden=512) @ 1000 tracks",
        "args": ["--max-tracks", "1000", "--output-dir", "outputs/midi/scale_crnn_l_1000",
                 "--epochs-train", "30", "--batch-size", "16", "--lr", "1e-3",
                 "--dropout", "0.3", "--weight-decay", "1e-4",
                 "--early-stop-patience", "15", "--use-mix",
                 "--hidden", "512"],
    },
]


def run_exp(exp):
    name, script, desc = exp["name"], exp["script"], exp["desc"]
    args = exp["args"]
    print(f"\n{'='*60}")
    print(f"  {name}: {desc}")
    print(f"{'='*60}")

    t0 = time.time()
    cmd = [PY, script] + args
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")

    elapsed = (time.time() - t0) / 60
    f1 = 0.0
    pit = 0.0
    for line in result.stdout.split("\n"):
        if "Best" in line and "F1=" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p.startswith("F1="):
                    f1 = float(p.split("=")[1].rstrip(","))
                if "pit_acc=" in p:
                    pit = float(p.split("=")[1].rstrip(","))

    # Fallback: try to parse from output
    if f1 == 0.0:
        for line in result.stdout.split("\n"):
            if "F1=" in line:
                try:
                    idx = line.index("F1=")
                    f1 = float(line[idx+3:].split()[0].rstrip(","))
                except:
                    pass

    print(f"  Result: F1={f1:.4f}, PitchAcc={pit:.4f}, Time={elapsed:.1f}min")
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
    return {"name": name, "f1": f1, "pit_acc": pit, "time": elapsed, "ok": result.returncode == 0}


def main():
    results = []
    # Previous results for comparison
    prev = {
        "HuBERT-ctx-200": 0.280,
        "HuBERT-mix-200": 0.289,
        "CRNN-ctx-200": 0.195,
        "CRNN-mix-200": 0.228,
        "Transformer-baseline": 0.130,
    }

    for exp in EXPERIMENTS:
        r = run_exp(exp)
        results.append(r)

    # Summary
    print(f"\n{'='*70}")
    print("SCALING EXPERIMENT RESULTS")
    print(f"{'='*70}")
    print(f"{'Experiment':<25} {'F1':>8} {'PitchAcc':>10} {'Time':>8}")
    print("-" * 55)

    all_results = {
        **prev,
        **{r["name"]: r["f1"] for r in results},
    }

    for name, f1 in sorted(all_results.items(), key=lambda x: -x[1]):
        marker = " ★" if f1 == max(all_results.values()) else ""
        print(f"{name:<25} {f1:>8.4f}{marker}")

    print(f"\nBest: {max(all_results, key=all_results.get)} = {max(all_results.values()):.4f}")
    print(f"Target: 0.80")


if __name__ == "__main__":
    main()
