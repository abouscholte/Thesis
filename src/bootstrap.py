from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score


def bootstrap_ap_difference(predictions, first, second, n_boot=1000, seed=42):
    dates = pd.to_datetime(predictions["FlightDate"]).dt.normalize()
    day_ids, days = pd.factorize(dates)
    actual = predictions["DepDel15"].to_numpy()
    first_scores = predictions[first].to_numpy()
    second_scores = predictions[second].to_numpy()

    estimate = (average_precision_score(actual, first_scores) - average_precision_score(actual, second_scores))

    rng = np.random.default_rng(seed)
    differences = []

    for _ in range(n_boot):
        sampled_days = rng.integers(0, len(days), size=len(days))
        weights = np.bincount(sampled_days, minlength=len(days))[day_ids]

        difference = (
            average_precision_score(actual, first_scores, sample_weight=weights)
            - average_precision_score(actual, second_scores, sample_weight=weights)
        )
        differences.append(difference)

    lower, upper = np.quantile(differences, [0.025, 0.975])

    return {
        "ap_difference": estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "bootstrap_repetitions": n_boot,
    }


def bootstrap_checkpoint(folder, name, prediction_files, first, second, n_boot=1000, seed=42):
    """Identify cached intervals by inputs, algorithm, comparison and settings."""
    def digest(path):
        checksum = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
        return checksum.hexdigest()

    signature = {
        "prediction_hashes": {label: digest(path) for label, path in sorted(prediction_files.items())},
        "algorithm_hash": digest(__file__),
        "first": first, "second": second,
        "n_boot": n_boot, "seed": seed,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / f"{name}_{fingerprint}.csv"
    metadata = checkpoint.with_suffix(".json")
    row = None
    if checkpoint.exists() and metadata.exists():
        if json.loads(metadata.read_text()) == signature:
            cached = pd.read_csv(checkpoint)
            assert len(cached) == 1
            assert cached["bootstrap_repetitions"].eq(n_boot).all()
            assert np.isfinite(cached[["ap_difference", "ci_lower", "ci_upper"]].to_numpy()).all()
            row = cached.iloc[0].to_dict()
    return checkpoint, signature, row


def save_bootstrap_checkpoint(checkpoint, signature, row):
    """Publish metadata last so interrupted writes are not reused."""
    temporary = checkpoint.with_suffix(".tmp")
    pd.DataFrame([row]).to_csv(temporary, index=False)
    temporary.replace(checkpoint)
    metadata = checkpoint.with_suffix(".json")
    temporary = metadata.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(signature, indent=2))
    temporary.replace(metadata)
