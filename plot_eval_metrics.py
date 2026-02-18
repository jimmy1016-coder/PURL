"""
Epoch 200~227에 대한 EER, minDCF 비교 플롯.
실험: NCU 1ep, 5ep, 30ep, Supcon
"""
import re
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

OUTPUTS = Path("outputs")
EXPERIMENTS = {
    "NCU (1ep)": "ncu-Softloss-alpha0.5-1ep",
    "NCU (5ep)": "ncu-Softloss-alpha0.5-5ep",
    "NCU (30ep)": "ncu-Softloss-alpha0.5-30ep",
    "Supcon": "supcon",
}
EPOCH_RANGE = (200, 228)  # 200~227 inclusive


def parse_metrics_file(filepath: Path) -> Optional[dict]:
    """Parse metrics file, return {eer: float, mindcf: float} or None if missing/invalid."""
    if not filepath.exists():
        return None
    try:
        text = filepath.read_text()
        eer_match = re.search(r"EER\s*=\s*([\d.]+)\s*%", text)
        dcf_match = re.search(r"minDCF\s*@\s*p=0\.05\s*=\s*([\d.]+)", text)
        if eer_match and dcf_match:
            return {"eer": float(eer_match.group(1)), "mindcf": float(dcf_match.group(1))}
    except Exception:
        pass
    return None


def load_experiment_metrics(exp_dir: str) -> Tuple[list, list, list]:
    """Load (epochs, eers, mindcfs) for epochs 200~227."""
    base = OUTPUTS / exp_dir / "evaluation_results"
    epochs, eers, mindcfs = [], [], []
    for ep in range(*EPOCH_RANGE):
        f = base / f"list_test_hard2_metrics_epoch{ep}.txt"
        m = parse_metrics_file(f)
        if m:
            epochs.append(ep)
            eers.append(m["eer"])
            mindcfs.append(m["mindcf"])
    return epochs, eers, mindcfs


def main():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]
    for (label, exp_dir), c in zip(EXPERIMENTS.items(), colors):
        epochs, eers, mindcfs = load_experiment_metrics(exp_dir)
        if not epochs:
            print(f"[SKIP] No data for {label} ({exp_dir})")
            continue
        ax1.plot(epochs, eers, "o-", label=label, color=c, markersize=4)
        ax2.plot(epochs, mindcfs, "o-", label=label, color=c, markersize=4)

    ax1.set_ylabel("EER (%)")
    ax1.set_title("EER comparison (epoch 200~227)")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("minDCF @ p=0.05")
    ax2.set_xlabel("Epoch")
    ax2.set_title("minDCF comparison (epoch 200~227)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUTS / "eer_mindcf_comparison_epoch200_227.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
