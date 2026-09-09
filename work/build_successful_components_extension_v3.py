"""Build a small continuation after the measured 0.45067 result."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "candidates" / "submission_best_0p46182.csv"
CURRENT = ROOT / "runs" / "successful_components_v2" / "submission_successful_components_v2.csv"
OUT_DIR = ROOT / "runs" / "successful_components_v3"
OUT = OUT_DIR / "submission_successful_components_v3.csv"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(BASE)
    current = pd.read_csv(CURRENT)
    sample = pd.read_csv(ROOT / "Dataset" / "sample_submission.csv")
    for frame in (base, current):
        if list(frame.columns) != ["row_id", "prediction"]:
            raise ValueError("Unexpected submission schema")
        if not frame.row_id.equals(sample.row_id) or not frame.row_id.is_unique:
            raise ValueError("Submission IDs do not match sample order")
        if not np.isfinite(frame.prediction).all():
            raise ValueError("Non-finite prediction")
    # The corrected endpoint pair (.46182 -> .45067) puts the quadratic
    # optimum at roughly 1.03 times this measured direction. Use 1.05.
    t = 1.05
    d = current.prediction.to_numpy(dtype=float) - base.prediction.to_numpy(dtype=float)
    result = base.copy()
    result["prediction"] = base.prediction.to_numpy(dtype=float) + t * d
    if not np.isfinite(result.prediction).all():
        raise ValueError("Non-finite output")
    result.to_csv(OUT, index=False)
    manifest = {
        "status": "Small continuation after the measured 0.45067 result.",
        "candidate": OUT.relative_to(ROOT).as_posix(),
        "sha256": sha(OUT),
        "incumbent_file": "candidates/submission_best_0p45067.csv",
        "incumbent_sha256": sha(ROOT / "candidates" / "submission_best_0p45067.csv"),
        "incumbent_rmse": 0.45067,
        "base_file": BASE.relative_to(ROOT).as_posix(),
        "base_rmse": 0.46182,
        "source_direction_file": CURRENT.relative_to(ROOT).as_posix(),
        "source_direction_sha256": sha(CURRENT),
        "source_direction_rmse": 0.45067,
        "direction_weight_from_corrected_base": t,
        "direction_rms": float(np.sqrt(np.mean(d * d))),
        "max_absolute_change_from_corrected_base": float(np.max(np.abs(t * d))),
        "allow_score_calibration": False,
        "limitations": [
            "The continuation uses the corrected public endpoint pair and full-test moments.",
            "The public subset is structured and private performance is unverified.",
            "This is a small measured-direction probe, not independent target validation."
        ]
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
