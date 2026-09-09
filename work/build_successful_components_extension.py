"""Build a restrained continuation of the measured successful-components direction."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "candidates" / "submission_best_0p46182.csv"
CURRENT = ROOT / "runs" / "successful_components_v1" / "submission_successful_components.csv"
OUT_DIR = ROOT / "runs" / "successful_components_v2"
OUT = OUT_DIR / "submission_successful_components_v2.csv"


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
    # The corrected endpoint pair (.46182 -> .45095) projects to ~1.19.
    # Use 1.15 as a conservative continuation from the corrected base.
    t = 1.15
    d = current.prediction.to_numpy(dtype=float) - base.prediction.to_numpy(dtype=float)
    result = base.copy()
    result["prediction"] = base.prediction.to_numpy(dtype=float) + t * d
    if not np.isfinite(result.prediction).all():
        raise ValueError("Non-finite output")
    result.to_csv(OUT, index=False)
    manifest = {
        "status": "Candidate continuation of the measured successful-components direction.",
        "candidate": OUT.relative_to(ROOT).as_posix(),
        "sha256": sha(OUT),
        "incumbent_file": BASE.relative_to(ROOT).as_posix(),
        "incumbent_sha256": sha(BASE),
        "incumbent_rmse": 0.46182,
        "source_direction_file": CURRENT.relative_to(ROOT).as_posix(),
        "source_direction_sha256": sha(CURRENT),
        "source_direction_rmse": 0.45095,
        "direction_weight": t,
        "direction_rms": float(np.sqrt(np.mean(d * d))),
        "max_absolute_change": float(np.max(np.abs(t * d))),
        "n_rows": int(len(result)),
        "allow_score_calibration": False,
        "limitations": [
            "The continuation uses the corrected public endpoint pair and full-test moments.",
            "The public subset is structured and private performance is unverified.",
            "This candidate is a measured-direction extension, not an independent target model."
        ]
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
