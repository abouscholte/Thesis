"""Joint calendar-day bootstrap for paired AP contrasts across airports.

Reads saved 2025 predictions only; no model is refitted. Each bootstrap
replicate resamples calendar units (single days or non-overlapping 7-day
blocks) once and applies the same weights to every airport and every model,
so cross-airport dependence (shared national disruption days) and the
pairing between models are both preserved. Intervals condition on the fitted
models and on these ten airports.
"""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

CONDITIONS = ["unseen", "reference"]
YEAR_START = np.datetime64("2025-01-01")


class FastAP:
    """Weighted average precision with sklearn's step definition.

    Sorting is done once; each call only needs a gather and two cumulative
    sums, which makes thousands of bootstrap replicates feasible.
    """

    def __init__(self, actual, scores):
        order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
        sorted_scores = np.asarray(scores, dtype=float)[order]
        self.order = order.astype(np.int32)
        self.actual = np.asarray(actual)[order].astype(np.int8)
        self.thresholds = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(order) - 1].astype(np.int32)

    def __call__(self, weights=None):
        w = np.ones(len(self.order)) if weights is None else np.asarray(weights, dtype=float)[self.order]
        tp = np.cumsum(self.actual * w)[self.thresholds]
        total = np.cumsum(w)[self.thresholds]
        with np.errstate(invalid="ignore", divide="ignore"):
            precision = np.where(total > 0, tp / total, 0.0)
        return float((np.diff(np.r_[0.0, tp]) * precision).sum() / tp[-1])


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_wide(project_root, manifest):
    """One frame per airport: labels, dates, constant and all 12 model columns."""
    project_root = Path(project_root)
    wide = {}
    for airport, group in manifest.groupby("test_airport"):
        base = None
        for entry in group.itertuples():
            frame = (pd.read_csv(project_root / entry.prediction_file, parse_dates=["FlightDate"])
                     .set_index("test_row_id").sort_index())
            if base is None:
                base = frame[["FlightDate", "DepDel15", "constant"]].copy()
            else:
                assert base.index.equals(frame.index)
                assert np.array_equal(base["DepDel15"], frame["DepDel15"])
                assert np.allclose(base["constant"], frame["constant"])
            for condition in CONDITIONS:
                base[f"{entry.family}|{entry.variant}|{condition}"] = frame[condition].to_numpy(float)
        assert base.index.is_unique
        wide[airport] = base
    return wide


def calendar_units(dates, scheme):
    """Day index (0-364), or non-overlapping 7-day blocks with 31 Dec in the last block."""
    day = (dates.to_numpy().astype("datetime64[D]") - YEAR_START).astype(int)
    assert day.min() >= 0 and day.max() <= 364
    if scheme == "day":
        return day, 365
    if scheme == "week":
        return np.minimum(day // 7, 51), 52
    raise ValueError(scheme)


def run_joint_bootstrap(wide, scheme, n_boot=1000, seed=42, progress=True):
    airports = sorted(wide)
    columns = ["constant"] + [c for c in wide[airports[0]].columns if "|" in c]
    scorers, units = {}, {}
    for airport in airports:
        frame = wide[airport]
        actual = frame["DepDel15"].to_numpy().astype(int)
        units[airport], n_units = calendar_units(frame["FlightDate"], scheme)
        for column in columns:
            scorers[airport, column] = FastAP(actual, frame[column].to_numpy(float))
    point = np.array([[scorers[a, c]() for c in columns] for a in airports])
    rng = np.random.default_rng(seed)
    replicates = np.empty((n_boot, len(airports), len(columns)))
    for b in range(n_boot):
        counts = np.bincount(rng.integers(0, n_units, n_units), minlength=n_units).astype(float)
        for i, airport in enumerate(airports):
            weights = counts[units[airport]]
            for j, column in enumerate(columns):
                replicates[b, i, j] = scorers[airport, column](weights)
        if progress and (b + 1) % 100 == 0:
            print(f"{scheme}: {b + 1}/{n_boot}", flush=True)
    return dict(point=point, replicates=replicates, airports=airports, columns=columns)


def cached_joint_bootstrap(project_root, manifest, wide, folder, scheme, n_boot=1000, seed=42):
    """Reuse a result only if predictions, this module, settings and versions match."""
    project_root, folder = Path(project_root), Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    signature = {
        "prediction_hashes": {f: file_hash(project_root / f) for f in sorted(manifest["prediction_file"])},
        "module_hash": file_hash(__file__),
        "scheme": scheme, "n_boot": n_boot, "seed": seed,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
    }
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    path = folder / f"joint_{scheme}_{fingerprint}.npz"
    metadata = path.with_suffix(".json")
    if path.exists() and metadata.exists() and json.loads(metadata.read_text()) == signature:
        stored = np.load(path, allow_pickle=False)
        print(f"Loaded: {path.name}")
        return dict(point=stored["point"], replicates=stored["replicates"],
                    airports=[str(a) for a in stored["airports"]], columns=[str(c) for c in stored["columns"]])
    result = run_joint_bootstrap(wide, scheme, n_boot, seed)
    temporary = path.with_name(path.stem + "_tmp.npz")
    np.savez(temporary, point=result["point"], replicates=result["replicates"],
             airports=np.array(result["airports"]), columns=np.array(result["columns"]))
    temporary.replace(path)
    metadata.write_text(json.dumps(signature, indent=2))
    return result


def contrast_definitions():
    """Name, contrast family (for multiplicity), first column, second column."""
    rows = []
    for family in ["logistic", "forest", "xgboost"]:
        rows.append((f"weather|{family}", "weather", f"{family}|weather|unseen", f"{family}|flight_only|unseen"))
        for variant in ["flight_only", "weather"]:
            rows.append((f"reference|{family}|{variant}", "reference",
                         f"{family}|{variant}|reference", f"{family}|{variant}|unseen"))
    for variant in ["flight_only", "weather"]:
        for first, second in [("xgboost", "logistic"), ("forest", "logistic"), ("xgboost", "forest")]:
            rows.append((f"{first}-{second}|{variant}", "model_family",
                         f"{first}|{variant}|unseen", f"{second}|{variant}|unseen"))
        rows.append((f"logistic-constant|{variant}", "baseline", f"logistic|{variant}|unseen", "constant"))
    return rows


def _percentile_p(samples):
    """Two-sided percentile-bootstrap p-value, floored at 1/B (then only an upper bound)."""
    p = 2 * min((samples <= 0).mean(), (samples >= 0).mean())
    return float(min(1.0, max(p, 1 / len(samples))))


def _wald_p(estimate, samples):
    """Two-sided normal-approximation p-value using the bootstrap standard error.

    Used for Holm adjustment because, unlike the percentile p-value, it is not
    limited by the 1/B resolution of a finite number of replicates.
    """
    se = float(np.std(samples, ddof=1))
    return 1.0 if se == 0 else float(math.erfc(abs(estimate) / se / math.sqrt(2)))


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def summarise(result, scheme):
    """Per-airport and equal-airport-mean contrasts with 95% percentile intervals."""
    columns, airports = result["columns"], result["airports"]
    point, reps = result["point"], result["replicates"]
    rows = []
    for name, group, first, second in contrast_definitions():
        i, j = columns.index(first), columns.index(second)
        estimate, samples = point[:, i] - point[:, j], reps[:, :, i] - reps[:, :, j]
        for k, airport in enumerate(airports):
            lower, upper = np.quantile(samples[:, k], [0.025, 0.975])
            rows.append(dict(contrast=name, group=group, airport=airport, scheme=scheme,
                             estimate=estimate[k], ci_lower=lower, ci_upper=upper,
                             bootstrap_se=float(np.std(samples[:, k], ddof=1)),
                             p_percentile=_percentile_p(samples[:, k]),
                             p_wald=_wald_p(estimate[k], samples[:, k])))
        mean_samples = samples.mean(axis=1)
        lower, upper = np.quantile(mean_samples, [0.025, 0.975])
        rows.append(dict(contrast=name, group=group, airport="Equal-airport mean", scheme=scheme,
                         estimate=estimate.mean(), ci_lower=lower, ci_upper=upper,
                         bootstrap_se=float(np.std(mean_samples, ddof=1)),
                         p_percentile=_percentile_p(mean_samples),
                         p_wald=_wald_p(estimate.mean(), mean_samples)))
    table = pd.DataFrame(rows)
    per_airport = table["airport"].ne("Equal-airport mean") & table["group"].ne("baseline")
    table["p_holm"] = np.nan
    for group, index in table.loc[per_airport].groupby("group").groups.items():
        table.loc[index, "p_holm"] = holm(table.loc[index, "p_wald"])
    table["n_boot"] = reps.shape[0]
    return table
