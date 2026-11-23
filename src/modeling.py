from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import sklearn
import xgboost

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

CATEGORICAL = ["Reporting_Airline", "Dest", "Month", "DayOfWeek", "departure_hour"]
NUMERIC = ["Distance"]
WEATHER = ["temperature", "dew_point_temperature", "wind_speed", "visibility"]
METRICS = ["average_precision", "roc_auc", "log_loss"]

def make_model(family, with_weather=False):
    numeric = NUMERIC + WEATHER if with_weather else NUMERIC

    numeric_prepreration = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()) if with_weather else StandardScaler()

    preprocessing = ColumnTransformer([("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                                       ("numeric", numeric_prepreration, numeric)])

    if family == "logistic":
        estimator = LogisticRegression(max_iter=1000)
    elif family == "forest":
        estimator = RandomForestClassifier(
            n_estimators=200,
            max_depth=16,
            min_samples_leaf=50,
            max_samples=0.3,
            n_jobs=-1,
            random_state=42,
        )
    elif family == "xgboost":
        estimator = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=4,
            random_state=42,
        )

    else:
        raise ValueError(f"Unkown model family: {family}")

    return make_pipeline(preprocessing, estimator), CATEGORICAL + numeric

CANDIDATES = {
    "logistic": [
        {"logisticregression__C": 0.1},
        {"logisticregression__C": 1.0},
        {"logisticregression__C": 10.0},
    ],
    "forest": [
        {"randomforestclassifier__max_depth": 10,
         "randomforestclassifier__min_samples_leaf": 100},
        {"randomforestclassifier__max_depth": 16,
         "randomforestclassifier__min_samples_leaf": 50},
        {"randomforestclassifier__max_depth": 24,
         "randomforestclassifier__min_samples_leaf": 20},
    ],
    "xgboost": [
        {"xgbclassifier__max_depth": 3},
        {"xgbclassifier__max_depth": 4},
        {"xgbclassifier__max_depth": 6},
    ],
}

def evaluate_airports(data, estimator, columns, test_airport):
    assert data["FlightDate"].dt.year.eq(2024).all()
    assert data["Origin"].nunique() == 10
    assert test_airport in set(data["Origin"])

    development = data.loc[data["Origin"].ne(test_airport)]
    scores = []

    for airport in sorted(development["Origin"].unique()):
        source = development.loc[development["Origin"].ne(airport) & development["FlightDate"].lt("2024-10-01")]
        target = development.loc[development["Origin"].eq(airport) & development["FlightDate"].ge("2024-10-01")]

        assert len(source) > 0 and len(target) > 0
        assert source["Origin"].nunique() == 8
        assert set(source["Origin"]).isdisjoint(target["Origin"])
        assert source["FlightDate"].max() < target["FlightDate"].min()

        model = clone(estimator)
        model.fit(source[columns], source["DepDel15"].astype(int))

        actual = target["DepDel15"].astype(int)
        probability = model.predict_proba(target[columns])[:, 1]

        scores.append({
            "test_airport": test_airport,
            "validation_airport": airport,
            "flights": len(target),
            "delay_rate": actual.mean(),
            "average_precision": average_precision_score(actual, probability),
            "roc_auc": roc_auc_score(actual, probability),
            "log_loss": log_loss(actual, probability),
        })

        print(f"{test_airport} held out - {airport} validated", flush=True)

    return pd.DataFrame(scores)

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def tune_airport(data, test_airport, output_dir, data_hash):
    output_dir = Path(output_dir) / test_airport
    output_dir.mkdir(parents=True, exist_ok=True)

    context = {
        "data": data_hash,
        "code": file_hash(__file__),
        "versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    expected_airports = set(data["Origin"]) - {test_airport}
    parts = []

    for family, candidates in CANDIDATES.items():
        for with_weather in [False, True]:
            variant = "weather" if with_weather else "flight_only"

            for candidate_id, settings in enumerate(candidates):
                estimator, columns = make_model(family, with_weather)
                estimator.set_params(**settings)

                signature = {
                    **context,
                    "test_airport": test_airport,
                    "family": family,
                    "variant": variant,
                    "candidate_id": candidate_id,
                    "settings": settings,
                    "columns": columns,
                }
                fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]

                name = f"{family}_{variant}_{candidate_id}_{fingerprint}"
                checkpoint = output_dir / f"{name}.csv"

                if checkpoint.exists():
                    scores = pd.read_csv(checkpoint)
                    print(f"Loaded: {name}", flush=True)
                else:
                    print(f"Running: {name}", flush=True)
                    scores = evaluate_airports(data, estimator, columns, test_airport)
                    scores = scores.assign(
                        family=family,
                        variant=variant,
                        candidate_id=candidate_id,
                        settings=json.dumps(settings, sort_keys=True)
                    )

                    temporary = checkpoint.with_suffix(".tmp")
                    scores.to_csv(temporary, index=False)
                    temporary.replace(checkpoint)

                    checkpoint.with_suffix(".json").write_text(json.dumps(signature, indent=2), encoding="utf-8")

                assert len(scores) == 9
                assert set(scores["validation_airport"]) == expected_airports
                assert scores["test_airport"].eq(test_airport).all()
                assert scores["family"].eq(family).all()
                assert scores["variant"].eq(variant).all()
                assert scores["candidate_id"].eq(candidate_id).all()
                assert np.isfinite(scores[METRICS].to_numpy()).all()

                parts.append(scores)

    return pd.concat(parts, ignore_index=True)

def fit_predict_pair(data, test_data, selection, output_dir):
    airport = selection["test_airport"]
    family = selection["family"]
    with_weather = selection["variant"] == "weather"
    settings = json.loads(selection["settings"])

    target = test_data.loc[test_data["Origin"].eq(airport)]
    unseen_train = data.loc[data["Origin"].ne(airport)]

    predictions = target[["FlightDate", "Origin", "DepDel15"]].copy()
    predictions["test_row_id"] = target.index
    predictions["constant"] = unseen_train["DepDel15"].mean()

    for name, training in [
        ("unseen", unseen_train),
        ("reference", data),
    ]:
        model, columns = make_model(family, with_weather)
        model.set_params(**settings)

        model.fit(training[columns], training["DepDel15"].astype(int))
        predictions[name] = model.predict_proba(target[columns])[:, 1]

        print(f"{airport}: {family} {name} fitted", flush=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{airport}_{family}_{selection['variant']}.csv.gz"
    predictions.to_csv(output_dir / filename, index=False)

    return predictions

def run_final_pair(data, test_data, selection, output_dir, data_hash, test_data_hash):
    signature = {
        "training_data": data_hash,
        "test_data": test_data_hash,
        "code": file_hash(__file__),
        "airport": selection["test_airport"],
        "family": selection["family"],
        "variant": selection["variant"],
        "settings": json.loads(selection["settings"]),
        "versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }

    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]

    folder = Path(output_dir) / fingerprint
    filename = f"{selection['test_airport']}_{selection['family']}_{selection['variant']}.csv.gz"
    checkpoint = folder / filename
    marker = folder / "complete.json"

    if checkpoint.exists() and marker.exists():
        predictions = pd.read_csv(checkpoint, parse_dates=["FlightDate"])
        print(f"Loaded: {filename}", flush=True)
    else:
        predictions = fit_predict_pair(data, test_data, selection, folder)

    target = test_data.loc[test_data["Origin"].eq(selection["test_airport"])]

    assert len(predictions) == len(target)
    assert predictions["test_row_id"].is_unique
    assert np.array_equal(predictions["test_row_id"], target.index)
    assert np.array_equal(predictions["DepDel15"], target["DepDel15"])

    probabilities = predictions[["constant", "unseen", "reference"]]
    assert np.isfinite(probabilities.to_numpy()).all()
    assert probabilities.ge(0).all().all()
    assert probabilities.le(1).all().all()

    marker.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    return checkpoint