from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CUTOFF = pd.Timestamp("2025-10-01")
NUMERIC_FEATURES = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
]
CATEGORICAL_FEATURES = ["pickup", "delivery", "equipment"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "posted_rate"


def prepare_features(frame: pd.DataFrame, weight_policy: str = "nonpositive-as-missing") -> pd.DataFrame:
    features = frame[FEATURES].copy()
    if weight_policy == "nonpositive-as-missing":
        features["weight"] = features["weight"].where(features["weight"] > 0)
    return features


def build_model(criterion: str = "squared_error") -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "regressor",
                RandomForestRegressor(
                    random_state=42,
                    n_jobs=-1,
                    criterion=criterion,
                ),
            ),
        ]
    )


def print_segment_errors(
    label: str,
    groups: pd.Series,
    actual: pd.Series,
    predicted: np.ndarray,
    minimum_rows: int = 1,
    limit: int | None = None,
) -> None:
    diagnostics = pd.DataFrame(
        {
            "segment": groups.astype(str).to_numpy(),
            "actual": actual.to_numpy(),
            "predicted": predicted,
        }
    )
    diagnostics["absolute_error"] = (diagnostics["predicted"] - diagnostics["actual"]).abs()
    diagnostics["signed_error"] = diagnostics["predicted"] - diagnostics["actual"]
    summary = (
        diagnostics.groupby("segment", dropna=False)
        .agg(
            n=("actual", "size"),
            mae=("absolute_error", "mean"),
            median_error=("signed_error", "median"),
        )
    )
    summary = summary.loc[summary["n"] >= minimum_rows].sort_values("mae", ascending=False)
    if limit is not None:
        summary = summary.head(limit)
    print(f"\n{label} (segments with n >= {minimum_rows}; worst MAE first):")
    print(summary.round(2).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a baseline on Jan-Sep development rows and score October."
    )
    parser.add_argument("--data", default="train-test.csv")
    parser.add_argument(
        "--weight-policy",
        choices=("nonpositive-as-missing", "raw"),
        default="nonpositive-as-missing",
        help="How to treat recorded nonpositive weight values before fit-only imputation.",
    )
    parser.add_argument(
        "--criterion",
        choices=("squared_error", "absolute_error"),
        default="squared_error",
        help="Random Forest split criterion; squared_error is the current benchmark.",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.data, parse_dates=["date"])
    fit_rows = data.loc[data["date"] < CUTOFF].copy()
    holdout_rows = data.loc[data["date"] >= CUTOFF].copy()

    fit_features = prepare_features(fit_rows, args.weight_policy)
    holdout_features = prepare_features(holdout_rows, args.weight_policy)
    features = FEATURES.copy()
    model = build_model(args.criterion)

    started = perf_counter()
    model.fit(fit_features, fit_rows[TARGET])
    fit_seconds = perf_counter() - started

    predictions = model.predict(holdout_features)
    holdout_mae = mean_absolute_error(holdout_rows[TARGET], predictions)
    errors = predictions - holdout_rows[TARGET].to_numpy()
    absolute_errors = np.abs(errors)
    constant_rate = fit_rows[TARGET].median()
    constant_predictions = np.full(len(holdout_rows), constant_rate)
    constant_mae = mean_absolute_error(holdout_rows[TARGET], constant_predictions)
    mae_improvement = constant_mae - holdout_mae

    print(f"Cutoff: {CUTOFF.date()}")
    print(f"Training rows: {len(fit_rows):,}")
    print(f"Holdout rows: {len(holdout_rows):,}")
    print(f"Features: {', '.join(features)}")
    print("Excluded: load_id, date, market_index, quote_signal")
    print(f"Weight policy: {args.weight_policy}; remaining missing values median-imputed from fit rows")
    print(f"Model: RandomForestRegressor(criterion={args.criterion!r}, random_state=42, n_jobs=-1)")
    print("Metric: October holdout MAE (lower is better)")
    print(f"Constant predictor: training median posted_rate = {constant_rate:.2f}")
    print(f"Constant predictor holdout MAE: {constant_mae:.2f}")
    print(f"Random forest holdout MAE: {holdout_mae:.2f}")
    print(f"MAE improvement vs constant: {mae_improvement:.2f}")
    print(f"Fit runtime: {fit_seconds:.2f} seconds")

    print("\nHoldout residual distribution (prediction - posted_rate):")
    print(
        pd.Series(errors).describe(percentiles=[0.01, 0.10, 0.50, 0.90, 0.99])
        .round(2)
        .to_string()
    )
    print(f"Overpredictions: {(errors > 0).sum():,}; underpredictions: {(errors < 0).sum():,}; exact: {(errors == 0).sum():,}")

    long_haul_cutoff = fit_rows["distance"].quantile(0.75)
    high_rate_cutoff = fit_rows[TARGET].quantile(0.95)
    print(
        f"\nFit-derived segment cutoffs: long-haul distance > Jan-Sep P75 "
        f"({long_haul_cutoff:.2f}); high-rate posted_rate > Jan-Sep P95 ({high_rate_cutoff:.2f})"
    )
    for label, mask in [
        ("long-haul", holdout_rows["distance"] > long_haul_cutoff),
        ("not long-haul", holdout_rows["distance"] <= long_haul_cutoff),
        ("high-rate", holdout_rows[TARGET] > high_rate_cutoff),
        ("not high-rate", holdout_rows[TARGET] <= high_rate_cutoff),
        (
            "long-haul and high-rate",
            (holdout_rows["distance"] > long_haul_cutoff)
            & (holdout_rows[TARGET] > high_rate_cutoff),
        ),
    ]:
        segment_errors = errors[mask.to_numpy()]
        segment_absolute_errors = absolute_errors[mask.to_numpy()]
        print(
            f"{label}: n={len(segment_errors):,}; MAE={segment_absolute_errors.mean():.2f}; "
            f"mean residual={segment_errors.mean():.2f}; "
            f"underprediction={(segment_errors < 0).mean():.1%}"
        )

    weight = holdout_rows["weight"]
    weight_segments = pd.Series("positive", index=holdout_rows.index)
    weight_segments.loc[weight.isna()] = "missing"
    weight_segments.loc[weight.notna() & (weight <= 0)] = "nonpositive"
    positive_weight = weight.where(weight > 0)
    weight_bins = pd.cut(
        positive_weight,
        bins=[0, 15000, 25000, 35000, np.inf],
        labels=["(0, 15k]", "(15k, 25k]", "(25k, 35k]", ">35k"],
        include_lowest=True,
    )
    weight_segments.loc[weight > 0] = weight_bins.loc[weight > 0].astype(str)

    distance_bins = pd.qcut(holdout_rows["distance"], q=4, duplicates="drop")
    rate_segments = pd.cut(
        holdout_rows[TARGET],
        bins=[-np.inf, 5000, 10000, np.inf],
        labels=["<=5k", ">5k to 10k", ">10k"],
    )
    weekday = holdout_rows["date"].dt.day_name()
    lane = holdout_rows["pickup"].astype(str) + " -> " + holdout_rows["delivery"].astype(str)

    print_segment_errors("Equipment", holdout_rows["equipment"], holdout_rows[TARGET], predictions)
    print_segment_errors("Weight quality / positive-weight bands", weight_segments, holdout_rows[TARGET], predictions)
    print_segment_errors("Road distance quartiles", distance_bins, holdout_rows[TARGET], predictions)
    print_segment_errors("Weekday", weekday, holdout_rows[TARGET], predictions)
    print_segment_errors("Posted-rate bands", rate_segments, holdout_rows[TARGET], predictions)
    print_segment_errors("Pickup", holdout_rows["pickup"], holdout_rows[TARGET], predictions, minimum_rows=50, limit=10)
    print_segment_errors("Delivery", holdout_rows["delivery"], holdout_rows[TARGET], predictions, minimum_rows=50, limit=10)
    print_segment_errors("Lane", lane, holdout_rows[TARGET], predictions, minimum_rows=20, limit=10)

    total_absolute_error = absolute_errors.sum()
    for threshold in (5000, 10000, 20000):
        tail = holdout_rows[TARGET].to_numpy() > threshold
        tail_error = absolute_errors[tail].sum()
        print(
            f"\nTarget > {threshold:,}: n={tail.sum():,}; MAE={absolute_errors[tail].mean() if tail.any() else float('nan'):.2f}; "
            f"share of total absolute error={tail_error / total_absolute_error:.1%}"
        )

    largest = holdout_rows[["load_id", "date", "pickup", "delivery", "distance", "equipment", "weight", TARGET]].copy()
    largest["predicted_rate"] = predictions
    largest["signed_error"] = errors
    largest["absolute_error"] = absolute_errors
    print("\nLargest absolute holdout errors:")
    print(largest.nlargest(10, "absolute_error").round({"distance": 1, TARGET: 2, "predicted_rate": 2, "signed_error": 2, "absolute_error": 2}).to_string(index=False))


if __name__ == "__main__":
    main()