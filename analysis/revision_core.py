"""Calendar-aligned REV2 core. Measurement caches are genuine archived EGMS.

This is a post-review retrospective analysis of previously explored products.
Only historical feature interpolation is permitted; the future target is an
actual acquisition at exactly 72 days. No completed-product processing
availability or independent physical truth is reconstructed.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PRODUCTS = ("amsterdam", "venice", "alps")
DISPLAY = {"amsterdam": "Amsterdam", "venice": "Venice", "alps": "Jura"}
MODELS = ("persistence", "mean3grid", "local_linear144", "trend_seasonal360",
          "ridge_level", "ridge_increment", "ridge_increment_duplicate")
FEATURES = ("delta12", "delta36", "delta72", "delta144", "roughness144")
GRID_DAYS = np.arange(-360, 1, 12, dtype=np.int64)
HORIZON_DAYS = 72
HISTORY_DAYS = 365
FIT_ORIGIN_START = np.datetime64("2021-01-01", "D")
FIT_TARGET_END = np.datetime64("2021-05-01", "D")
CAL_ORIGIN_START = np.datetime64("2021-05-01", "D")
CAL_TARGET_END = np.datetime64("2022-10-01", "D")
TEST_ORIGIN_START = np.datetime64("2022-11-01", "D")
PENALTY = 0.1
BLOCK_SIZE = 2000
BOOTSTRAPS = 1000
SEED = 20261011
COORD_COLUMNS = ("easting", "northing", "latitude", "longitude")
REQUIRED_NUMERIC_FIELDS = ("latitude", "longitude", "easting", "northing", "rmse_ts",
                           "temporal_coherence", "mean_velocity", "los_east", "los_north",
                           "los_up", "gnss_velocity")
PRODUCT_FILES = {
    "amsterdam": "EGMS_L2b_088_0312_IW1_VV_2020_2024_1.csv",
    "venice": "EGMS_L2b_117_0266_IW3_VV_2020_2024_1.csv",
    "alps": "EGMS_L2b_088_0278_IW2_VV_2020_2024_1.csv",
}
EXPECTED_ROWS = {"amsterdam": 338161, "venice": 64133, "alps": 32678}
EXPECTED_HASHES = {
    "amsterdam": "EC0B9BC1879A85E4962327170060091BEE2ECA1ED001A6A9D82829BE1CD05E62",
    "venice": "BF57107E27A8E8CA604C4FC3E50A9520BF39EEE3B8188E948D46A4FE0E471E40",
    "alps": "015FFEC4AF6D51E82669A64BE57F6EC69D10F3960A9E3A568AC32945CDB297B2",
}


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False), encoding="utf-8")
    part.replace(path)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def date_array(tokens):
    return np.asarray([f"{s[:4]}-{s[4:6]}-{s[6:]}" for s in tokens], dtype="datetime64[D]")


def calendar_plan(dates):
    dates = np.asarray(dates, dtype="datetime64[D]")
    days = dates.astype(np.int64)
    assert len(np.unique(days)) == len(days) and np.all(np.diff(days) > 0)
    target_day = days + HORIZON_DAYS
    candidate = np.searchsorted(days, target_day)
    exact = candidate < len(days)
    exact[exact] &= days[candidate[exact]] == target_day[exact]
    calendar_candidate = dates >= FIT_ORIGIN_START
    history = days - days[0] >= HISTORY_DAYS
    eligible = calendar_candidate & history & exact
    origins = np.flatnonzero(eligible).astype(np.int64)
    targets = candidate[eligible].astype(np.int64)
    if not len(origins):
        raise ValueError("No origins have 365-day history and an exact 72-day acquisition")
    desired = days[origins, None] + GRID_DAYS[None, :]
    right = np.searchsorted(days, desired)
    exact_grid = days[right] == desired
    left = np.where(exact_grid, right, right - 1)
    assert np.min(left) >= 0
    assert np.all(right <= origins[:, None]), "Feature interpolation uses a future observation"
    brackets = days[right] - days[left]
    weight = np.divide(desired - days[left], brackets, out=np.zeros_like(desired, dtype=np.float64),
                       where=brackets != 0)
    assert np.all(right[:, -1] == origins) and np.all(left[:, -1] == origins)
    od, td = dates[origins], dates[targets]
    phases = {
        "fit": (od >= FIT_ORIGIN_START) & (td < FIT_TARGET_END),
        "calibration": (od >= CAL_ORIGIN_START) & (td < CAL_TARGET_END),
        "test": od >= TEST_ORIGIN_START,
    }
    assert np.all(sum(mask.astype(np.int64) for mask in phases.values()) <= 1)
    assert np.all(td - od == np.timedelta64(HORIZON_DAYS, "D"))
    assert np.all(td[phases["fit"]] < CAL_ORIGIN_START)
    assert np.all(td[phases["calibration"]] < TEST_ORIGIN_START)
    interpolation = {
        "grid_days_relative_to_origin": GRID_DAYS.tolist(),
        "interpolated_grid_fraction_by_origin": (~exact_grid).mean(axis=1).tolist(),
        "max_bracket_days_by_origin": brackets.max(axis=1).tolist(),
        "mean_interpolated_grid_fraction": float((~exact_grid).mean()),
        "maximum_bracket_days": int(brackets.max()),
        "origin_endpoint_is_observed": True,
        "future_observations_used_for_features": False,
    }
    for phase, mask in phases.items():
        interpolation[phase] = {
            "grid_fraction_interpolated": float((~exact_grid[mask]).mean()),
            "maximum_bracket_days": int(brackets[mask].max()),
        }
    dropout = {
        "acquisition_origins_on_or_after_2021_01_01": int(calendar_candidate.sum()),
        "insufficient_365_day_history": int((calendar_candidate & ~history).sum()),
        "missing_exact_72_day_target_after_history_check": int((calendar_candidate & history & ~exact).sum()),
        "eligible_origins": int(len(origins)),
        "eligible_origins_outside_evaluation_phases": int((~np.logical_or.reduce(list(phases.values()))).sum()),
    }
    return {"origins": origins, "targets": targets, "left": left, "right": right,
            "weight": weight, "phases": phases, "dates": dates,
            "interpolation": interpolation, "dropout": dropout}


def history_grid(series, plan, start, stop):
    """Float64 linear interpolation using observations at or before each origin."""
    left = np.asarray(series[start:stop, plan["left"]], dtype=np.float64)
    right = np.asarray(series[start:stop, plan["right"]], dtype=np.float64)
    return left + (right - left) * plan["weight"][None, :, :]


def grid_features(grid):
    level = grid[:, :, -1]
    delta = [level - grid[:, :, -1 - lag // 12] for lag in (12, 36, 72, 144)]
    roughness = np.diff(grid[:, :, -13:], axis=2).std(axis=2, ddof=0)
    return np.stack([*delta, roughness], axis=2)


def linear_weights(days, seasonal=False):
    days = np.asarray(days, dtype=np.float64)
    design = np.column_stack([np.ones(len(days)), days])
    future = np.asarray([1., HORIZON_DAYS])
    if seasonal:
        design = np.column_stack([design, np.sin(2 * np.pi * days / 365.25),
                                  np.cos(2 * np.pi * days / 365.25)])
        future = np.r_[future, np.sin(2 * np.pi * HORIZON_DAYS / 365.25),
                       np.cos(2 * np.pi * HORIZON_DAYS / 365.25)]
    return future @ np.linalg.pinv(design)


LINEAR144_WEIGHTS = linear_weights(GRID_DAYS[-13:])
SEASONAL360_WEIGHTS = linear_weights(GRID_DAYS, seasonal=True)


def fixed_predictions(grid):
    return {"persistence": grid[:, :, -1],
            "mean3grid": grid[:, :, -3:].mean(axis=2),
            "local_linear144": grid[:, :, -13:] @ LINEAR144_WEIGHTS,
            "trend_seasonal360": grid @ SEASONAL360_WEIGHTS}


def atomic_memmap(path, shape, dtype=np.float32):
    path = Path(path)
    return np.lib.format.open_memmap(path.with_suffix(".part.npy"), mode="w+", dtype=dtype, shape=shape)


def finish_memmap(array, path):
    array.flush()
    # Close explicitly before rename on Windows.
    array._mmap.close()
    Path(path).with_suffix(".part.npy").replace(path)


def resolve_raw(provenance, raw_dir=None):
    recorded = Path(provenance.get("csv", provenance.get("csv_filename", "")))
    if raw_dir is None:
        if recorded.is_file():
            return recorded
        raise FileNotFoundError("Original CSV is absent; supply --raw-dir containing identifier CSVs")
    matches = list(Path(raw_dir).rglob(recorded.name))
    if len(matches) != 1:
        raise ValueError(f"Expected one {recorded.name}; found {len(matches)}")
    return matches[0]


def read_provenance(root, name):
    path = Path(root) / "analysis" / "cache" / f"{name}_provenance.json"
    if path.is_file():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("sha256") != EXPECTED_HASHES[name]:
            raise ValueError(f"Unexpected raw-data version in {name} provenance")
        if Path(result.get("csv", result.get("csv_filename", ""))).name != PRODUCT_FILES[name]:
            raise ValueError(f"Product identifier mismatch in {name} provenance")
        return result
    return {"name": name, "csv": PRODUCT_FILES[name], "sha256": EXPECTED_HASHES[name],
            "valid_rows_loaded": EXPECTED_ROWS[name], "dropped_rows": 0}


def ensure_series_cache(root, name, raw, provenance, observed_hash):
    """Create a D: working array from CSV or verify its raw/date/array fingerprint.

    Primary originals have no incomplete rows. An unexpected incomplete field
    fails explicitly instead of silently changing the revision population.
    Large working arrays are never required to ship with the code package.
    """
    cache = Path(root) / "analysis" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    dates = [str(s) for s in pd.read_csv(raw, nrows=0, encoding="utf-8-sig").columns
             if str(s).isdigit() and len(str(s)) == 8]
    actual_days = date_array(dates)
    assert len(np.unique(actual_days)) == len(actual_days) and np.all(np.diff(actual_days.astype(int)) > 0)
    if "dates" in provenance:
        assert dates == provenance["dates"], "Cached provenance date list differs from original CSV"
    series_path = cache / f"{name}_series.npy"
    manifest_path = cache / f"{name}_series_cache_manifest.json"
    expected_shape = (EXPECTED_ROWS[name], len(dates))
    fingerprint = None
    if manifest_path.exists():
        fingerprint = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif (cache / "core" / name / "metadata.json").exists():
        previous = json.loads((cache / "core" / name / "metadata.json").read_text(encoding="utf-8"))
        if "series_cache_sha256" in previous:
            fingerprint = {"raw_sha256": previous["raw_sha256"],
                           "series_sha256": previous["series_cache_sha256"],
                           "shape": previous["cached_series_shape"],
                           "dates": [s.replace("-", "") for s in previous["dates"]]}
    if series_path.exists() and fingerprint is not None:
        assert fingerprint["raw_sha256"] == observed_hash
        assert fingerprint["dates"] == dates and tuple(fingerprint["shape"]) == expected_shape
        assert sha256_file(series_path) == fingerprint["series_sha256"], "Working-array fingerprint mismatch"
        series = np.load(series_path, mmap_mode="r")
        assert series.shape == expected_shape and series.dtype == np.float32
        for start in range(0, len(series), 8192):
            assert np.isfinite(series[start:start + 8192]).all()
    else:
        # Reading all required numeric fields establishes the complete-row rule
        # even though product-wide quality values never enter prediction.
        columns = ["pid", *REQUIRED_NUMERIC_FIELDS, *dates]
        table = pd.read_csv(raw, usecols=columns, encoding="utf-8-sig", engine="c",
                            dtype={"pid": "str", **{s: np.float32 for s in columns if s != "pid"}})
        assert len(table) == EXPECTED_ROWS[name]
        assert table["pid"].notna().all() and not table["pid"].duplicated().any()
        for column in columns[1:]:
            if not np.isfinite(table[column].to_numpy()).all():
                raise ValueError(f"Incomplete numeric {name} CSV field {column}; primary population would change")
        values = table[dates].to_numpy(dtype=np.float32, copy=False)
        if series_path.exists():
            series = np.load(series_path, mmap_mode="r")
            assert series.shape == expected_shape and series.dtype == np.float32
            for start in range(0, len(series), 8192):
                np.testing.assert_array_equal(series[start:start + 8192], values[start:start + 8192],
                                              err_msg="Unfingerprinted working array differs from original CSV")
        else:
            working = atomic_memmap(series_path, expected_shape)
            for start in range(0, len(values), 8192):
                working[start:start + 8192] = values[start:start + 8192]
            finish_memmap(working, series_path)
        fingerprint = {"raw_sha256": observed_hash, "series_sha256": sha256_file(series_path),
                       "shape": expected_shape, "dates": dates,
                       "all_required_numeric_fields_complete": True,
                       "unique_nonmissing_pid": True, "source": "original CSV full-field verification"}
        write_json(manifest_path, fingerprint)
    return {**provenance, "dates": dates, "valid_rows_loaded": EXPECTED_ROWS[name], "dropped_rows": 0}, fingerprint


def build_region_cache(root, name, raw_dir=None, chunk_points=1024):
    root = Path(root)
    cache = root / "analysis" / "cache"
    dest = cache / "core" / name
    dest.mkdir(parents=True, exist_ok=True)
    provenance = read_provenance(root, name)
    raw = resolve_raw(provenance, raw_dir)
    observed_hash = sha256_file(raw)
    if observed_hash != provenance["sha256"]:
        raise ValueError(f"Raw CSV hash mismatch for {name}; cached series cannot be reused")
    provenance, series_fingerprint = ensure_series_cache(root, name, raw, provenance, observed_hash)
    series = np.load(cache / f"{name}_series.npy", mmap_mode="r")
    assert series.shape == (provenance["valid_rows_loaded"], len(provenance["dates"]))
    assert provenance["dropped_rows"] == 0, "REV2 primary products must retain all source rows"
    table = pd.read_csv(raw, usecols=["pid", *COORD_COLUMNS],
                        dtype={"pid": "str", **{s: np.float64 for s in COORD_COLUMNS}},
                        encoding="utf-8-sig", engine="c")
    assert len(table) == len(series) and table["pid"].notna().all()
    assert not table["pid"].duplicated().any()
    coords = table[list(COORD_COLUMNS)].to_numpy(dtype=np.float64)
    assert np.isfinite(coords).all()
    np.save(dest / "coords.npy", coords)
    raw_blocks = np.floor(coords[:, :2] / BLOCK_SIZE).astype(np.int64)
    keys, labels = np.unique(raw_blocks, axis=0, return_inverse=True)
    draws = np.random.default_rng(SEED + PRODUCTS.index(name) * 100).integers(
        0, len(keys), size=(BOOTSTRAPS, len(keys)), dtype=np.int64)
    np.save(dest / "block_keys_2000.npy", keys)
    np.save(dest / "block_labels_2000.npy", labels.astype(np.int64))
    np.save(dest / "bootstrap_draws_2000.npy", draws)
    plan = calendar_plan(date_array(provenance["dates"]))
    for key in ("origins", "targets"):
        np.save(dest / f"{key}.npy", plan[key])
    for phase, mask in plan["phases"].items():
        np.save(dest / f"phase_{phase}.npy", mask)
    shape = (series.shape[0], len(plan["origins"]))
    paths = {s: dest / f"{s}.npy" for s in ("level", "target", "roughness", "features")}
    arrays = {s: atomic_memmap(p, (*shape, 5) if s == "features" else shape)
              for s, p in paths.items()}
    pred_paths = {model: dest / f"pred_amsterdam_{model}.npy" for model in MODELS[:4]}
    predictions = {model: atomic_memmap(p, shape) for model, p in pred_paths.items()}
    for start in range(0, shape[0], chunk_points):
        stop = min(start + chunk_points, shape[0])
        grid = history_grid(series, plan, start, stop)
        features = grid_features(grid)
        arrays["features"][start:stop] = features
        arrays["level"][start:stop] = grid[:, :, -1]
        arrays["target"][start:stop] = series[start:stop, plan["targets"]]
        arrays["roughness"][start:stop] = features[:, :, -1]
        for model, pred in fixed_predictions(grid).items():
            predictions[model][start:stop] = pred
    for s, a in arrays.items():
        finish_memmap(a, paths[s])
    for s, a in predictions.items():
        finish_memmap(a, pred_paths[s])
    dates = plan["dates"]
    phases = {}
    for phase, mask in plan["phases"].items():
        od, td = dates[plan["origins"][mask]], dates[plan["targets"][mask]]
        phases[phase] = {"origin_count": int(mask.sum()), "point_origin_pairs": int(shape[0] * mask.sum()),
                         "first_origin": str(od[0]), "last_origin": str(od[-1]),
                         "first_target": str(td[0]), "last_target": str(td[-1]),
                         "lead_days_unique": [HORIZON_DAYS]}
    metadata = {
        "region": name, "display_name": DISPLAY[name], "csv_filename": raw.name,
        "raw_sha256": observed_hash, "raw_sha_matches_original_audit": True,
        "cached_series_shape": list(series.shape), "series_cache_sha256": series_fingerprint["series_sha256"],
        "all_valid_points_retained": True, "points": int(shape[0]), "eligible_origin_count": int(shape[1]),
        "array_order": "point, eligible origin; features add a final feature axis",
        "coordinate_columns": COORD_COLUMNS, "coordinate_dtype": "float64 original CSV values",
        "feature_columns": FEATURES, "dates": [str(d) for d in dates],
        "origin_dates": [str(d) for d in dates[plan["origins"]]],
        "target_dates": [str(d) for d in dates[plan["targets"]]],
        "origins": plan["origins"], "targets": plan["targets"], "phases": phases,
        "interpolation": plan["interpolation"], "dropout": plan["dropout"],
        "block_size_m": BLOCK_SIZE, "block_count": len(keys), "bootstrap_replicates": BOOTSTRAPS,
        "post_review_analysis_of_previously_explored_products": True,
        "forecast_target_is_real_acquisition_not_interpolated": True,
        "processing_hindsight_remains": True,
        "bbox_lon_lat": [float(coords[:, 3].min()), float(coords[:, 3].max()),
                         float(coords[:, 2].min()), float(coords[:, 2].max())],
        "cache_ready": True,
    }
    write_json(dest / "metadata.json", metadata)
    return metadata


def load_core_region(root, name, mmap_mode="r"):
    dest = Path(root) / "analysis" / "cache" / "core" / name
    data = {path.stem: np.load(path, mmap_mode=mmap_mode) for path in dest.glob("*.npy")
            if not path.name.endswith(".part.npy")}
    data["meta"] = json.loads((dest / "metadata.json").read_text(encoding="utf-8"))
    return data


def model_features(region, model, start, stop, mask=None):
    x = np.asarray(region["features"][start:stop], dtype=np.float64)
    if mask is not None:
        x = x[:, mask]
    if model == "ridge_level":
        level = np.asarray(region["level"][start:stop], dtype=np.float64)
        if mask is not None:
            level = level[:, mask]
        x = np.concatenate([level[:, :, None], x], axis=2)
    elif model == "ridge_increment_duplicate":
        x = np.concatenate([x, x[:, :, 3:4] / 12.], axis=2)
    elif model != "ridge_increment":
        raise ValueError(f"Unsupported fitted model {model}")
    return x


def fit_ridge(region, model, chunk_points=4096):
    mask = region["phase_fit"]
    dimension = 5 if model == "ridge_increment" else 6
    n = int(region["level"].shape[0] * mask.sum())
    sx = np.zeros(dimension); sx2 = np.zeros(dimension); sy = 0.
    for start in range(0, len(region["level"]), chunk_points):
        stop = min(start + chunk_points, len(region["level"]))
        x = model_features(region, model, start, stop, mask).reshape(-1, dimension)
        y = np.asarray(region["target"][start:stop, mask], dtype=np.float64)
        if model != "ridge_level":
            y = y - region["level"][start:stop, mask].astype(np.float64)
        sx += x.sum(axis=0); sx2 += np.square(x).sum(axis=0); sy += y.sum()
    mean, ymean = sx / n, sy / n
    sd = np.sqrt(np.maximum(sx2 / n - mean ** 2, 0))
    sd[sd < 1e-8] = 1.
    xtx = np.zeros((dimension, dimension)); xty = np.zeros(dimension)
    for start in range(0, len(region["level"]), chunk_points):
        stop = min(start + chunk_points, len(region["level"]))
        x = model_features(region, model, start, stop, mask).reshape(-1, dimension)
        z = (x - mean) / sd
        y = np.asarray(region["target"][start:stop, mask], dtype=np.float64)
        if model != "ridge_level":
            y = y - region["level"][start:stop, mask].astype(np.float64)
        xtx += z.T @ z; xty += z.T @ (y.reshape(-1) - ymean)
    beta = np.linalg.solve(xtx / n + PENALTY * np.eye(dimension), xty / n)
    coefficients = beta / sd
    columns = (["level", *FEATURES] if model == "ridge_level" else
               [*FEATURES, "delta144_div12_duplicate"] if model.endswith("duplicate") else list(FEATURES))
    return {"model": model, "source": region["meta"]["region"], "fit_pairs": n,
            "feature_columns": columns, "means": mean, "population_sds": sd,
            "standardized_coefficients": beta, "target_mean_unpenalized_intercept": ymean,
            "raw_unit_coefficients": dict(zip(columns, coefficients)),
            "raw_unit_intercept": float(ymean - mean @ coefficients),
            "penalty": PENALTY, "loss": "mean squared error plus lambda times squared standardized coefficients",
            "fit_phase": region["meta"]["phases"]["fit"], "population_sd_ddof": 0}


def predict_ridge_chunk(region, fitted, start, stop, level_shift=0.):
    model = fitted["model"]
    x = model_features(region, model, start, stop)
    if model == "ridge_level":
        x[:, :, 0] += level_shift
    y = (x - np.asarray(fitted["means"])) / np.asarray(fitted["population_sds"])
    pred = y @ np.asarray(fitted["standardized_coefficients"]) + fitted["target_mean_unpenalized_intercept"]
    if model != "ridge_level":
        pred += region["level"][start:stop].astype(np.float64) + level_shift
    return pred


def cache_ridge_prediction(root, region, fitted, chunk_points=4096):
    dest = Path(root) / "analysis" / "cache" / "core" / region["meta"]["region"]
    path = dest / f"pred_{fitted['source']}_{fitted['model']}.npy"
    array = atomic_memmap(path, region["level"].shape)
    for start in range(0, len(array), chunk_points):
        stop = min(start + chunk_points, len(array))
        array[start:stop] = predict_ridge_chunk(region, fitted, start, stop)
    finish_memmap(array, path)
    return path


def exact_radius(values, alpha=.10):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite calibration residual")
    k = math.ceil((len(values) + 1) * (1 - alpha))
    if k > len(values):
        raise ValueError("Calibration sample too small for a finite exact radius")
    return float(np.partition(values, k - 1)[k - 1])


def point_error_sums(prediction, target, mask, chunk_points=4096):
    n = len(prediction)
    sums = {key: np.empty(n, dtype=np.float64) for key in ("mae", "mse", "bias")}
    for start in range(0, n, chunk_points):
        stop = min(start + chunk_points, n)
        error = prediction[start:stop, mask].astype(np.float64) - target[start:stop, mask].astype(np.float64)
        sums["mae"][start:stop] = np.abs(error).sum(axis=1)
        sums["mse"][start:stop] = np.square(error).sum(axis=1)
        sums["bias"][start:stop] = error.sum(axis=1)
    return sums


def cluster_summary(point_sums, labels, draws, origin_count):
    count = len(draws[0])
    denominators = np.bincount(labels, minlength=count).astype(np.float64) * origin_count
    bootstrap_denominators = denominators[draws].sum(axis=1)
    n = int(denominators.sum())
    values, cis = {}, {}
    for key, point_sum in point_sums.items():
        block_sum = np.bincount(labels, weights=point_sum, minlength=count)
        empirical = block_sum.sum() / n
        bootstrap = block_sum[draws].sum(axis=1) / bootstrap_denominators
        if key == "mse":
            key, empirical, bootstrap = "rmse", math.sqrt(empirical), np.sqrt(bootstrap)
        values[key] = float(empirical)
        cis[key] = np.quantile(bootstrap, (.025, .975), method="linear").tolist()
    return {"n_pairs": n, "metrics": values, "ci_95_spatial_2000m": cis,
            "occupied_blocks": count, "bootstrap_replicates": len(draws),
            "pair_weighted": True, "all_origins_retained_with_each_point": True}


def synthetic_check():
    """Explicitly synthetic assertions. Never writes empirical result files."""
    dates = np.arange(np.datetime64("2020-01-01"), np.datetime64("2025-01-01"), np.timedelta64(12, "D"))
    plan = calendar_plan(dates)
    t = (dates - dates[0]).astype(int)
    series = np.vstack([.3 * t + 7, -.2 * t - 3]).astype(np.float64)
    grid = history_grid(series, plan, 0, 2)
    for model in ("local_linear144", "trend_seasonal360"):
        np.testing.assert_allclose(fixed_predictions(grid)[model], series[:, plan["targets"]], atol=1e-9)
    np.testing.assert_allclose(grid[:, :, -1], series[:, plan["origins"]])
    assert np.all(plan["right"] <= plan["origins"][:, None])
    perturb = series.copy()
    origin = int(plan["origins"][0])
    perturb[:, origin + 1:] += 10000
    np.testing.assert_allclose(history_grid(perturb, plan, 0, 2)[:, 0], grid[:, 0])
    modified_dates = np.delete(dates, plan["targets"][0])
    modified = calendar_plan(modified_dates)
    assert dates[origin] not in modified_dates[modified["origins"]]
    # Deliberate irregular past dates exercise interpolation without any target interpolation.
    irregular = np.delete(dates, 4)
    p = calendar_plan(irregular)
    assert p["interpolation"]["mean_interpolated_grid_fraction"] > 0
    np.testing.assert_allclose(history_grid(np.vstack([.3 * (irregular - dates[0]).astype(int) + 7]), p, 0, 1),
                               .3 * ((irregular[p["origins"]] - dates[0]).astype(int)[None, :, None] + GRID_DAYS) + 7,
                               atol=1e-10)
    assert exact_radius(np.arange(1, 20)) == 18
    rng = np.random.default_rng(44)
    n, m = 17, len(plan["origins"])
    features = rng.normal(size=(n, m, 5)); level = rng.normal(size=(n, m))
    target = level + features @ np.arange(1, 6) + rng.normal(size=(n, m))
    region = {"features": features, "level": level, "target": target,
              "phase_fit": plan["phases"]["fit"], "meta": {"region": "synthetic", "phases": {"fit": {}}}}
    for model in MODELS[4:]:
        fit = fit_ridge(region, model, chunk_points=7)
        x = model_features(region, model, 0, n, region["phase_fit"]).reshape(-1, len(fit["means"]))
        y = target[:, region["phase_fit"]]
        if model != "ridge_level":
            y = y - level[:, region["phase_fit"]]
        z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=0)
        b = np.linalg.solve(z.T @ z / len(z) + PENALTY * np.eye(z.shape[1]),
                            z.T @ (y.reshape(-1) - y.mean()) / len(z))
        np.testing.assert_allclose(b, fit["standardized_coefficients"], atol=1e-12)
        base = predict_ridge_chunk(region, fit, 0, n)
        shifted = predict_ridge_chunk(region, fit, 0, n, level_shift=10)
        expected = 10 * (fit["raw_unit_coefficients"]["level"] if model == "ridge_level" else 1)
        np.testing.assert_allclose(shifted - base, expected, atol=1e-12)
    return {"explicitly_synthetic": True, "passed": True,
            "checks": ["exact72 targets", "calendar purging", "past-only irregular interpolation",
                       "future feature invariance", "OLS affine recovery", "closed-form ridge",
                       "constant-reference shift", "exact radius order statistic"]}
