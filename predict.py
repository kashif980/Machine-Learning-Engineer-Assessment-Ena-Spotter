from __future__ import annotations

import numpy as np
import pandas as pd

from baseline import FEATURES, TARGET, build_model, prepare_features

DEVELOPMENT_PATH = "train-test.csv"
VALIDATION_PATH = "validation.csv"
TEMPLATE_PATH = "validation-predictions-template.csv"
DECEMBER_PATH = "december-chart-inputs.csv"
VALIDATION_OUTPUT_PATH = "validation_predictions.csv"


def development_coordinate_map(development: pd.DataFrame) -> pd.DataFrame:
    pickup_coordinates = development[
        ["pickup", "pickup_lat", "pickup_lon"]
    ].rename(columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"})
    delivery_coordinates = development[
        ["delivery", "delivery_lat", "delivery_lon"]
    ].rename(columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"})
    coordinates = pd.concat(
        [pickup_coordinates, delivery_coordinates], ignore_index=True
    ).drop_duplicates()
    conflicts = coordinates.groupby("city")[["lat", "lon"]].nunique().gt(1).any(axis=1)
    if conflicts.any():
        cities = ", ".join(conflicts[conflicts].index.astype(str))
        raise ValueError(f"development data has conflicting coordinates for: {cities}")
    return coordinates.drop_duplicates("city").set_index("city")


def add_december_coordinates(
    december: pd.DataFrame, coordinate_map: pd.DataFrame
) -> pd.DataFrame:
    result = december.copy()
    for side, latitude, longitude in [
        ("pickup", "pickup_lat", "pickup_lon"),
        ("delivery", "delivery_lat", "delivery_lon"),
    ]:
        result[latitude] = result[side].map(coordinate_map["lat"])
        result[longitude] = result[side].map(coordinate_map["lon"])
    coordinate_columns = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
    if result[coordinate_columns].isna().any().any():
        missing_cities = pd.concat(
            [
                result.loc[result["pickup_lat"].isna(), "pickup"],
                result.loc[result["delivery_lat"].isna(), "delivery"],
            ]
        ).unique()
        raise ValueError(
            "December cities lack coordinates in development data: "
            + ", ".join(map(str, missing_cities))
        )
    return result


def main() -> None:
    development = pd.read_csv(DEVELOPMENT_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    template = pd.read_csv(TEMPLATE_PATH)
    december = pd.read_csv(DECEMBER_PATH)

    if development[TARGET].isna().any():
        raise ValueError("development target contains missing posted_rate values")
    if validation["load_id"].duplicated().any() or template["load_id"].duplicated().any():
        raise ValueError("validation/template load_id values must be unique")
    if set(validation["load_id"]) != set(template["load_id"]):
        raise ValueError("validation IDs do not match prediction template IDs")

    model = build_model(criterion="squared_error")
    model.fit(
        prepare_features(development),
        development[TARGET],
    )

    validation_rates = model.predict(prepare_features(validation))
    if not np.isfinite(validation_rates).all() or (validation_rates <= 0).any():
        raise ValueError("validation predictions must be finite and positive")
    rates_by_id = pd.Series(validation_rates, index=validation["load_id"])
    validation_output = template[["load_id"]].copy()
    validation_output["predicted_rate"] = validation_output["load_id"].map(rates_by_id)
    if validation_output["predicted_rate"].isna().any():
        raise ValueError("some template IDs did not receive predictions")
    validation_output.to_csv(VALIDATION_OUTPUT_PATH, index=False)

    coordinate_map = development_coordinate_map(development)
    december_with_coordinates = add_december_coordinates(december, coordinate_map)
    december_rates = model.predict(prepare_features(december_with_coordinates))
    if not np.isfinite(december_rates).all() or (december_rates <= 0).any():
        raise ValueError("December predictions must be finite and positive")
    december["predicted_rate"] = december_rates
    december.to_csv(DECEMBER_PATH, index=False)

    print(f"Created {VALIDATION_OUTPUT_PATH}: {len(validation_output):,} rows")
    print(
        f"Filled {DECEMBER_PATH}: {len(december):,} rows; "
        "the baseline excludes date, so the fixed-input daily predictions are identical."
    )


if __name__ == "__main__":
    main()