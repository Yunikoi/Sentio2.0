"""Export trained ``RandomForestClassifier`` joblib bundle to Core ML ``.mlmodel``."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .row_windows import ROW_BASE_NUMERIC_COLUMN_NAMES


def export_rf_joblib_to_coreml(
    joblib_path: Path,
    out_dir: Path,
    *,
    target_name: str = "fall",
    window_samples: Optional[int] = None,
    hop_samples: Optional[int] = None,
) -> Tuple[Path, Path]:
    """
    Load ``{"model": clf, "feature_cols": [...]}`` from joblib and write Core ML artifacts.

    coremltools may disable sklearn conversion for newer scikit-learn; this function
    temporarily enables the internal flag used by Apple's converter.
    """
    import joblib

    import coremltools._deps as deps

    deps._HAS_SKLEARN = True
    import coremltools.converters.sklearn._random_forest_classifier as rfc

    importlib.reload(rfc)

    bundle: dict[str, Any] = joblib.load(joblib_path)
    clf = bundle["model"]
    feature_cols: List[str] = list(bundle["feature_cols"])

    mlmodel = rfc.convert(clf, feature_cols, target_name)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_model = out_dir / "FallDetectorRowRF.mlmodel"
    mlmodel.save(str(out_model))

    manifest: dict[str, Any] = {
        "mlmodel": str(out_model.as_posix()),
        "target_output": target_name,
        "probability_output": "classProbability",
        "feature_names_in_order": feature_cols,
        "n_features": len(feature_cols),
        "class_labels": [int(x) for x in getattr(clf, "classes_", [])],
        "base_numeric_columns": list(ROW_BASE_NUMERIC_COLUMN_NAMES),
    }
    if window_samples is not None:
        manifest["window_samples"] = int(window_samples)
    if hop_samples is not None:
        manifest["hop_samples"] = int(hop_samples)
    manifest_path = out_dir / "FallDetectorRowRF_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return out_model, manifest_path
