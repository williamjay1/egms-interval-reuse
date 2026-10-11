"""Calendar-aligned calibration, component controls and selection diagnostics.

Read-only inputs are revision_core's memory-mapped arrays.  This script never
retrains a predictor or changes the point/target population.  Run --self-test
first, then run without that flag once all core prediction files are complete.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

PRODUCTS = ("amsterdam", "venice", "alps")
METRICS = ("mae", "rmse_squared", "bias_prediction_minus_target", "coverage",
           "mean_width", "interval_score", "lower_tail", "upper_tail")
ALPHA = .10
K_VALUES = (6, 12, 24)
SELECTION_QUANTILES = (.5, .7, .8, .9, .95)
BOOTSTRAPS = 1000
CHUNK = 16384
SEED = 20261011


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def process_memory():
    """Windows OS-maintained working-set peak, not an inferred array estimate."""
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("page_faults", ctypes.c_ulong),
                    *[(key, ctypes.c_size_t) for key in
                      ("peak_working_set", "working_set", "quota_peak_paged", "quota_paged",
                       "quota_peak_nonpaged", "quota_nonpaged", "pagefile_usage",
                       "peak_pagefile_usage", "private_usage")]]
    try:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        item = Counters()
        item.cb = ctypes.sizeof(item)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(item), item.cb):
            raise OSError(ctypes.get_last_error())
        return {"measurement": "Windows GetProcessMemoryInfo",
                "working_set_bytes": int(item.working_set),
                "os_peak_working_set_bytes": int(item.peak_working_set),
                "private_commit_bytes": int(item.private_usage),
                "os_peak_pagefile_usage_bytes": int(item.peak_pagefile_usage)}
    except (AttributeError, OSError) as error:
        return {"measurement": "unavailable", "reason": str(error)}


def order_radius(values, alpha=ALPHA):
    """Exact corrected residual order statistic, without a coverage claim."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("Calibration residuals must be complete and finite")
    rank = math.ceil((array.size + 1) * (1 - alpha))
    if rank > array.size:
        return float("inf")
    array.partition(rank - 1)
    return float(array[rank - 1])


def mature_columns(origin_dates, target_dates, now, k):
    """Recent eligible origin dates whose labels strictly precede now."""
    eligible = np.flatnonzero((origin_dates >= np.datetime64("2021-05-01"))
                              & (target_dates < now))
    if eligible.size < k:
        raise ValueError(f"Only {eligible.size} mature dates for K={k} at {now}")
    return eligible[-k:]


def decomposition(all_n, all_hits, kept_n, kept_hits):
    all_n, all_hits, kept_n, kept_hits = [np.asarray(x, dtype=np.float64)
                                        for x in (all_n, all_hits, kept_n, kept_hits)]
    if all_n.sum() <= 0 or kept_n.sum() <= 0:
        raise ValueError("Coverage decomposition needs both populations")
    w_all, w_kept = all_n / all_n.sum(), kept_n / kept_n.sum()
    c_all = np.divide(all_hits, all_n, out=np.zeros_like(all_hits), where=all_n > 0)
    c_kept = np.divide(kept_hits, kept_n, out=np.zeros_like(kept_hits), where=kept_n > 0)
    composition = float(np.sum((w_kept - w_all) * c_all))
    within = float(np.sum(w_kept * (c_kept - c_all)))
    total = float(kept_hits.sum() / kept_n.sum() - all_hits.sum() / all_n.sum())
    if not np.isclose(composition + within, total, atol=2e-14, rtol=0):
        raise AssertionError("Composition and within-group terms do not add up")
    return {"coverage_change": total, "composition_change": composition,
            "within_group_change": within, "identity_error": composition + within - total,
            "all_weights": w_all, "kept_weights": w_kept,
            "all_group_coverage": c_all, "kept_group_coverage": c_kept}


@dataclass
class Region:
    name: str
    folder: Path
    meta: dict
    origin_dates: np.ndarray
    target_dates: np.ndarray
    origin_indices: np.ndarray
    target_indices: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    target: np.ndarray
    roughness: np.ndarray
    coordinates: np.ndarray
    blocks: np.ndarray
    block_keys: np.ndarray
    draws: np.ndarray

    @classmethod
    def load(cls, root, name):
        folder = root / "analysis" / "cache" / "core" / name
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8-sig"))
        def array(stem):
            return np.load(folder / f"{stem}.npy", mmap_mode="r")
        origins = np.asarray(meta["origin_dates"], dtype="datetime64[D]")
        targets = np.asarray(meta["target_dates"], dtype="datetime64[D]")
        cal = np.flatnonzero(array("phase_calibration"))
        test = np.flatnonzero(array("phase_test"))
        if not np.all((targets - origins).astype(int) == 72):
            raise AssertionError("Main task contains a non-72-day pair")
        if not np.all((origins[cal] >= np.datetime64("2021-05-01"))
                      & (targets[cal] < np.datetime64("2022-10-01"))):
            raise AssertionError("Static calibration violates the frozen dates")
        if not np.all(origins[test] >= np.datetime64("2022-11-01")):
            raise AssertionError("A test origin precedes the test cutoff")
        blocks = array("block_labels_2000")
        keys = array("block_keys_2000")
        draws = array("bootstrap_draws_2000")
        if draws.shape != (BOOTSTRAPS, len(keys)):
            raise AssertionError("Expected shared 1000-replicate spatial draws")
        if set(np.unique(blocks)) != set(range(len(keys))):
            raise AssertionError("Cell labels must be contiguous occupied-cell indices")
        target = array("target")
        roughness = array("roughness")
        if target.shape != roughness.shape or target.shape[1] != len(origins):
            raise AssertionError("Core target/roughness shapes disagree")
        return cls(name, folder, meta, origins, targets, array("origins"), array("targets"),
                   cal, test, target, roughness, array("coords"), blocks, keys, draws)

    def prediction(self, source):
        pred = np.load(self.folder / f"pred_{source}_ridge_increment.npy", mmap_mode="r")
        if pred.shape != self.target.shape:
            raise AssertionError("Prediction centers and labels do not align")
        return pred


class Radii:
    """Cache each unique predictor/source/window calibration calculation."""
    def __init__(self, regions):
        self.regions = regions
        self.values = {}
        self.records = []
        source = regions["amsterdam"]
        rough = np.asarray(source.roughness[:, source.calibration], dtype=np.float64)
        self.cuts = np.quantile(rough, [1/3, 2/3])
        self.thresholds = dict(zip(SELECTION_QUANTILES,
                                   np.quantile(rough, SELECTION_QUANTILES)))
        del rough

    def get(self, predictor_source, calibration_source, columns, grouped=False):
        columns = np.asarray(columns, dtype=np.int64)
        key = (predictor_source, calibration_source, tuple(columns), grouped)
        if key in self.values:
            return self.values[key]
        region = self.regions[calibration_source]
        prediction = region.prediction(predictor_source)
        errors = np.subtract(prediction[:, columns], region.target[:, columns],
                             dtype=np.float64)
        np.abs(errors, out=errors)
        if grouped:
            if calibration_source != "amsterdam":
                raise ValueError("Grouped experiment has source-Amsterdam calibration only")
            rough = np.asarray(region.roughness[:, columns])
            groups = np.searchsorted(self.cuts, rough, side="right")
            radius = np.asarray([order_radius(errors[groups == i]) for i in range(3)])
            group_counts = [int(np.sum(groups == i)) for i in range(3)]
            del groups, rough
        else:
            radius = order_radius(errors)
            group_counts = []
        record = {"calibration_id": len(self.records), "predictor_source": predictor_source,
                  "calibration_source": calibration_source, "grouped": grouped,
                  "n_dates": len(columns), "n_pairs": int(errors.size),
                  "first_origin": str(region.origin_dates[columns[0]]),
                  "last_origin": str(region.origin_dates[columns[-1]]),
                  "first_target": str(region.target_dates[columns[0]]),
                  "last_target": str(region.target_dates[columns[-1]]),
                  "column_indices": columns.tolist(), "radius_mm": jsonable(radius),
                  "group_pair_counts": group_counts}
        result = (radius, record)
        self.values[key] = result
        self.records.append(record)
        del errors
        gc.collect()
        return result


def estimates(sums, count):
    sums = np.asarray(sums, dtype=np.float64)
    count = np.asarray(count, dtype=np.float64)
    values = np.divide(sums, count[..., None], out=np.full_like(sums, np.nan),
                       where=count[..., None] > 0)
    values[..., 1] = np.sqrt(values[..., 1])
    return values


def equal_cell_estimates(sums, count):
    """Equal weight per nonempty cell, then take the root of weighted MSE."""
    sums = np.asarray(sums, dtype=np.float64)
    count = np.asarray(count, dtype=np.float64)
    cell_means = np.divide(sums, count[..., None], out=np.full_like(sums, np.nan),
                           where=count[..., None] > 0)
    values = np.nanmean(cell_means, axis=-2)
    values[..., 1] = np.sqrt(values[..., 1])
    return values


def temporal_draws(n, length, rng):
    """Moving contiguous observed-origin blocks; no independence assertion."""
    if n < length:
        raise ValueError("Test period shorter than requested time block")
    starts = rng.integers(0, n - length + 1, size=(BOOTSTRAPS, math.ceil(n / length)))
    return (starts[..., None] + np.arange(length)).reshape(BOOTSTRAPS, -1)[:, :n]


@dataclass
class Evaluation:
    protocol: str
    target_name: str
    predictor_source: str
    n: int
    cell_count: np.ndarray
    cell_sums: np.ndarray
    date_count: np.ndarray
    date_sums: np.ndarray
    cell_date_count: np.ndarray
    cell_date_outside: np.ndarray
    group_n: np.ndarray
    group_hits: np.ndarray
    summary: dict
    per_date: list


def evaluate(region, predictor_source, radii, protocol, cuts, threshold=None):
    """One date and a bounded point chunk at a time; shared point/date pairs."""
    started = time.monotonic()
    prediction = region.prediction(predictor_source)
    b, t, m = len(region.block_keys), len(region.test), len(METRICS)
    count = np.zeros((b, t), dtype=np.int64)
    sums = np.zeros((b, t, m), dtype=np.float64)
    outside_count = np.zeros((b, t), dtype=np.int64)
    group_n = np.zeros(3, dtype=np.int64)
    group_hits = np.zeros(3, dtype=np.int64)
    per_date = []
    for j, col in enumerate(region.test):
        radius, calibration_record = radii[j]
        for first in range(0, prediction.shape[0], CHUNK):
            last = min(first + CHUNK, prediction.shape[0])
            err = np.subtract(prediction[first:last, col], region.target[first:last, col],
                              dtype=np.float64)
            rough = np.asarray(region.roughness[first:last, col])
            groups = np.searchsorted(cuts, rough, side="right")
            keep = np.ones(len(err), dtype=bool) if threshold is None else rough <= threshold
            labels = np.asarray(region.blocks[first:last])[keep]
            errors = err[keep]
            absolute = np.abs(errors)
            rad = np.asarray(radius)[groups[keep]] if np.ndim(radius) else float(radius)
            hits = absolute <= rad
            lower, upper = errors > rad, errors < -rad
            values = (absolute, errors**2, errors, hits, np.broadcast_to(2*rad, errors.shape),
                      2*rad + (2/ALPHA)*np.maximum(absolute-rad, 0), lower, upper)
            count[:, j] += np.bincount(labels, minlength=b).astype(np.int64)
            outside_count[:, j] += np.bincount(labels, weights=~hits, minlength=b).astype(np.int64)
            for k, value in enumerate(values):
                sums[:, j, k] += np.bincount(labels, weights=value, minlength=b)
            group_n += np.bincount(groups[keep], minlength=3).astype(np.int64)
            group_hits += np.bincount(groups[keep], weights=hits, minlength=3).astype(np.int64)
        date_n = int(count[:, j].sum())
        value = estimates(sums[:, j].sum(axis=0), date_n)
        cell_value = equal_cell_estimates(sums[:, j], count[:, j])
        row = {"protocol": protocol, "target": region.name, "predictor_source": predictor_source,
               "origin_date": str(region.origin_dates[col]),
               "target_date": str(region.target_dates[col]), "lead_days": 72,
               "core_column": int(col), "origin_index": int(region.origin_indices[col]),
               "target_index": int(region.target_indices[col]), "n_pairs": date_n,
               "n_cells_with_output": int(np.sum(count[:, j] > 0)),
               "radius_mm": float(radius) if np.ndim(radius) == 0 else "",
               "calibration_id": calibration_record["calibration_id"],
               "calibration_n_dates": calibration_record["n_dates"],
               "calibration_n_pairs": calibration_record["n_pairs"],
               "calibration_first_origin": calibration_record["first_origin"],
               "calibration_last_origin": calibration_record["last_origin"],
               "calibration_last_target": calibration_record["last_target"]}
        for k, name in enumerate(METRICS):
            row[name.replace("rmse_squared", "rmse")] = float(value[k])
            row["equal_cell_" + name.replace("rmse_squared", "rmse")] = float(cell_value[k])
        row["outside_per_1000_pairs"] = 1000*(1-value[3])
        row["outside_relative_increase_over_nominal"] = (1-value[3]) / ALPHA - 1
        row["n_outside_pairs"] = int(outside_count[:, j].sum())
        row["n_lower_tail_pairs"] = int(sums[:, j, 6].sum())
        row["n_upper_tail_pairs"] = int(sums[:, j, 7].sum())
        row["pair_retention"] = date_n / prediction.shape[0]
        per_date.append(row)
    cell_n, cell_sums = count.sum(axis=1), sums.sum(axis=1)
    date_n, date_sums = count.sum(axis=0), sums.sum(axis=0)
    total_n = int(cell_n.sum())
    point_values = estimates(cell_sums.sum(axis=0), total_n)
    cell_values = equal_cell_estimates(cell_sums, cell_n)
    spatial_values = estimates(cell_sums[np.asarray(region.draws)].sum(axis=1),
                               cell_n[np.asarray(region.draws)].sum(axis=1))
    equal_cell_spatial = equal_cell_estimates(cell_sums[np.asarray(region.draws)],
                                             cell_n[np.asarray(region.draws)])
    summary = {"protocol": protocol, "target": region.name, "predictor_source": predictor_source,
               "n_pairs": total_n, "n_points": prediction.shape[0], "n_test_dates": t,
               "n_cells": b, "n_cells_with_output": int(np.sum(cell_n > 0)),
               "first_test_origin": str(region.origin_dates[region.test[0]]),
               "last_test_origin": str(region.origin_dates[region.test[-1]]),
               "first_test_target": str(region.target_dates[region.test[0]]),
               "last_test_target": str(region.target_dates[region.test[-1]]),
               "pair_retention": total_n / (prediction.shape[0]*t),
               "cell_mean_pair_retention": float(np.mean(cell_n / (np.bincount(region.blocks, minlength=b)*t))),
               "roughness_threshold": threshold, "elapsed_seconds": time.monotonic()-started}
    for k, name in enumerate(METRICS):
        key = name.replace("rmse_squared", "rmse")
        summary[key] = float(point_values[k])
        summary["equal_cell_" + key] = float(cell_values[k])
        lo, hi = np.quantile(spatial_values[:, k], [.025, .975])
        summary[key + "_spatial_lo"] = float(lo)
        summary[key + "_spatial_hi"] = float(hi)
        lo, hi = np.quantile(equal_cell_spatial[:, k], [.025, .975])
        summary["equal_cell_" + key + "_spatial_lo"] = float(lo)
        summary["equal_cell_" + key + "_spatial_hi"] = float(hi)
    summary["outside_per_1000_pairs"] = 1000*(1-point_values[3])
    summary["outside_relative_increase_over_nominal"] = (1-point_values[3])/ALPHA - 1
    summary["n_outside_pairs"] = int(outside_count.sum())
    summary["n_lower_tail_pairs"] = int(cell_sums[:, 6].sum())
    summary["n_upper_tail_pairs"] = int(cell_sums[:, 7].sum())
    for length in (6, 12):
        draw = temporal_draws(t, length, np.random.default_rng(SEED + PRODUCTS.index(region.name)*100 + length))
        values = estimates(date_sums[draw].sum(axis=1), date_n[draw].sum(axis=1))
        for k, name in enumerate(METRICS):
            lo, hi = np.quantile(values[:, k], [.025, .975])
            key = name.replace("rmse_squared", "rmse")
            summary[f"{key}_timeblock{length}_lo"] = float(lo)
            summary[f"{key}_timeblock{length}_hi"] = float(hi)
    return Evaluation(protocol, region.name, predictor_source, total_n, cell_n, cell_sums,
                      date_n, date_sums, count, outside_count, group_n, group_hits,
                      summary, per_date)


def paired_comparison(region, newer, reference):
    row = {"target": region.name, "protocol": newer.protocol, "reference": reference.protocol,
           "n_pairs_new": newer.n, "n_pairs_reference": reference.n,
           "delta_n_outside_pairs": newer.summary["n_outside_pairs"]-reference.summary["n_outside_pairs"],
           "delta_outside_per_1000_pairs": newer.summary["outside_per_1000_pairs"]-reference.summary["outside_per_1000_pairs"]}
    draw = np.asarray(region.draws)
    new_spatial = estimates(newer.cell_sums[draw].sum(axis=1), newer.cell_count[draw].sum(axis=1))
    old_spatial = estimates(reference.cell_sums[draw].sum(axis=1), reference.cell_count[draw].sum(axis=1))
    new_equal = equal_cell_estimates(newer.cell_sums[draw], newer.cell_count[draw])
    old_equal = equal_cell_estimates(reference.cell_sums[draw], reference.cell_count[draw])
    for k, name in enumerate(METRICS):
        key = name.replace("rmse_squared", "rmse")
        row["delta_" + key] = newer.summary[key] - reference.summary[key]
        lo, hi = np.quantile(new_spatial[:, k] - old_spatial[:, k], [.025, .975])
        row[f"delta_{key}_spatial_lo"], row[f"delta_{key}_spatial_hi"] = float(lo), float(hi)
        row["delta_equal_cell_" + key] = newer.summary["equal_cell_"+key] - reference.summary["equal_cell_"+key]
        lo, hi = np.quantile(new_equal[:, k]-old_equal[:, k], [.025, .975])
        row[f"delta_equal_cell_{key}_spatial_lo"] = float(lo)
        row[f"delta_equal_cell_{key}_spatial_hi"] = float(hi)
    for length in (6, 12):
        td = temporal_draws(len(region.test), length,
                            np.random.default_rng(SEED + PRODUCTS.index(region.name)*100 + length))
        new_time = estimates(newer.date_sums[td].sum(axis=1), newer.date_count[td].sum(axis=1))
        old_time = estimates(reference.date_sums[td].sum(axis=1), reference.date_count[td].sum(axis=1))
        for k, name in enumerate(METRICS):
            key = name.replace("rmse_squared", "rmse")
            lo, hi = np.quantile(new_time[:, k]-old_time[:, k], [.025, .975])
            row[f"delta_{key}_timeblock{length}_lo"] = float(lo)
            row[f"delta_{key}_timeblock{length}_hi"] = float(hi)
    return row


def autocorrelation_rows(evaluation, max_lag=12):
    values = evaluation.date_sums[:, 3] / evaluation.date_count
    rows = []
    for lag in range(1, min(max_lag, len(values)-2)+1):
        a, b = values[:-lag], values[lag:]
        corr = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else float("nan")
        rows.append({"protocol": evaluation.protocol, "target": evaluation.target_name,
                     "observed_origin_lag": lag, "n_date_pairs": len(a),
                     "coverage_autocorrelation": float(corr)})
    return rows


def cell_rows(region, evaluation, base=None, selection_quantile=None):
    point_count = np.bincount(region.blocks, minlength=len(region.block_keys))
    b, t = evaluation.cell_date_count.shape
    lat = np.bincount(region.blocks, weights=region.coordinates[:, 2], minlength=b) / point_count
    lon = np.bincount(region.blocks, weights=region.coordinates[:, 3], minlength=b) / point_count
    result = []
    for cell in range(b):
        counts = evaluation.cell_date_count[cell]
        has_output = counts > 0
        row = {"protocol": evaluation.protocol, "target": region.name, "cell_index": cell,
               "cell_easting_index": int(region.block_keys[cell, 0]),
               "cell_northing_index": int(region.block_keys[cell, 1]), "grid_m": 2000,
               "mean_latitude": float(lat[cell]), "mean_longitude": float(lon[cell]),
               "measurement_points_in_cell": int(point_count[cell]),
               "n_test_dates": t, "n_dates_with_output": int(has_output.sum()),
               "fraction_dates_with_output": float(has_output.mean()),
               "never_retained": bool(not has_output.any()),
               "at_most_25pct_dates_with_output": bool(has_output.mean() <= .25),
               "n_pairs_kept": int(counts.sum()), "n_pairs_available": int(point_count[cell]*t),
               "pair_retention_in_cell": float(counts.sum() / (point_count[cell]*t)),
               "mean_date_kept_fraction": float(np.mean(counts/point_count[cell])),
               "n_outside_pairs": int(evaluation.cell_date_outside[cell].sum()),
               "selection_quantile": selection_quantile}
        result.append(row)
    return result


def cell_date_rows(region, evaluation, selection_quantile=None):
    points = np.bincount(region.blocks, minlength=len(region.block_keys))
    result = []
    for j, col in enumerate(region.test):
        for cell in range(len(points)):
            n = int(evaluation.cell_date_count[cell, j])
            outside = int(evaluation.cell_date_outside[cell, j])
            result.append({"protocol": evaluation.protocol, "target": region.name,
                           "origin_date": str(region.origin_dates[col]), "target_date": str(region.target_dates[col]),
                           "cell_index": cell, "cell_easting_index": int(region.block_keys[cell, 0]),
                           "cell_northing_index": int(region.block_keys[cell, 1]),
                           "n_pairs_available": int(points[cell]), "n_pairs_kept": n,
                           "kept_fraction": n/points[cell], "all_rejected_on_date": n == 0,
                           "n_outside_pairs": outside, "outside_fraction": outside/n if n else None,
                           "selection_quantile": selection_quantile})
    return result


def repeated_radii(region, item):
    return [item] * len(region.test)


def self_test():
    origins = np.asarray(["2021-04-30", "2021-05-01", "2021-05-13", "2021-05-25"], dtype="datetime64[D]")
    targets = np.asarray(["2021-07-11", "2021-07-12", "2021-07-24", "2021-08-05"], dtype="datetime64[D]")
    chosen = mature_columns(origins, targets, np.datetime64("2021-08-05"), 2)
    assert chosen.tolist() == [1, 2]  # pre-May origin and equal-date immature label excluded.
    assert order_radius(np.arange(1., 11.), alpha=.2) == 9
    d = decomposition([50, 50, 0], [40, 30, 0], [40, 10, 0], [36, 4, 0])
    assert np.isclose(d["composition_change"], .06) and np.isclose(d["within_group_change"], .04)
    centers = np.asarray([0., 3., 7.])
    labels = np.asarray([1., 1., 4.])
    err = centers-labels
    mae_before = np.mean(np.abs(err))
    for radius in [.5, 2., 10.]:
        hits = np.abs(err) <= radius
        assert np.mean(np.abs(centers-labels)) == mae_before
        assert np.array_equal(centers, centers.copy())
        assert int(hits.sum()) + int(((err > radius)|(err < -radius)).sum()) == len(err)
    # Exercise the production evaluator, rather than merely copying centers.
    origin_dates = np.datetime64("2022-11-01") + np.arange(12)*np.timedelta64(12, "D")
    target = np.zeros((4, 12), dtype=np.float32)
    pred = np.broadcast_to(np.asarray([-1., 1.5, -3., 4.])[:, None], target.shape)
    fake = SimpleNamespace(name="amsterdam", origin_dates=origin_dates,
        target_dates=origin_dates+np.timedelta64(72, "D"),
        origin_indices=np.arange(12), target_indices=np.arange(12)+6,
        test=np.arange(12), target=target,
        roughness=np.broadcast_to(np.arange(4)[:, None], target.shape),
        block_keys=np.asarray([[0, 0], [1, 0]]), blocks=np.asarray([0, 0, 1, 1]),
        draws=np.random.default_rng(11).integers(0, 2, (BOOTSTRAPS, 2)),
        prediction=lambda source: pred)
    rec = {"calibration_id": 0, "n_dates": 20, "n_pairs": 80,
           "first_origin": "2021-05-01", "last_origin": "2022-07-01",
           "last_target": "2022-09-11"}
    small = evaluate(fake, "amsterdam", [(1., rec)]*12, "synthetic_q1", [1.5, 2.5])
    wide = evaluate(fake, "amsterdam", [(4., rec)]*12, "synthetic_q4", [1.5, 2.5])
    assert small.n == wide.n == 48
    assert small.summary["mae"] == wide.summary["mae"] == 2.375
    assert small.summary["coverage"] == .25 and wide.summary["coverage"] == 1
    assert small.summary["lower_tail"] == .5 and small.summary["upper_tail"] == .25
    assert np.isclose(small.summary["equal_cell_rmse"], np.sqrt(np.mean(pred**2)))
    return {"maturity_strict_before": "passed", "pre_calibration_cutoff_excluded": "passed",
            "exact_order_statistic": "passed", "selection_decomposition": "passed",
            "calibration_keeps_prediction_centers": "passed in production evaluator",
            "equal_cell_rmse_root_after_weighting": "passed", "tail_identity": "passed"}


def run(root):
    run_start = time.monotonic()
    start_utc = datetime.now(timezone.utc).isoformat()
    out = root / "results" / "revision_calibration"
    out.mkdir(parents=True, exist_ok=True)
    checks = self_test()
    regions = {name: Region.load(root, name) for name in PRODUCTS}
    q = Radii(regions)
    source = regions["amsterdam"]
    global_full = q.get("amsterdam", "amsterdam", source.calibration)
    grouped_full = q.get("amsterdam", "amsterdam", source.calibration, grouped=True)
    summaries, date_rows, comparisons, acfs = [], [], [], []
    composition_rows, group_rows, availability, availability_dates = [], [], [], []
    calibration_usage = []
    for name, region in regions.items():
        evaluations = {}
        schedules = {"source_full_global": repeated_radii(region, global_full),
                     "source_full_grouped": repeated_radii(region, grouped_full)}
        for k in K_VALUES:
            schedules[f"source_last{k}_global"] = repeated_radii(
                region, q.get("amsterdam", "amsterdam", source.calibration[-k:]))
            schedules[f"target_last{k}_global"] = repeated_radii(
                region, q.get("amsterdam", name, region.calibration[-k:]))
            rolling = []
            for col in region.test:
                now = region.origin_dates[col]
                columns = mature_columns(region.origin_dates, region.target_dates, now, k)
                if not np.all(region.target_dates[columns] < now):
                    raise AssertionError("An immature residual entered rolling calibration")
                rolling.append(q.get("amsterdam", name, columns))
            schedules[f"target_rolling{k}_global"] = rolling
        # Fixed Amsterdam centers, changing only the calibration population.
        for cal_source, other in regions.items():
            schedules[f"calibration_{cal_source}_full_global"] = repeated_radii(
                region, q.get("amsterdam", cal_source, other.calibration))
            schedules[f"calibration_{cal_source}_last12_global"] = repeated_radii(
                region, q.get("amsterdam", cal_source, other.calibration[-12:]))
        for protocol, schedule in schedules.items():
            print(f"[{name}] {protocol}", flush=True)
            result = evaluate(region, "amsterdam", schedule, protocol, q.cuts)
            evaluations[protocol] = result
            summaries.append(result.summary)
            date_rows.extend(result.per_date)
            acfs.extend(autocorrelation_rows(result))
            for j, (_, record) in enumerate(schedule):
                calibration_usage.append({"protocol": protocol, "target": name,
                                          "test_origin": str(region.origin_dates[region.test[j]]),
                                          "calibration_id": record["calibration_id"]})
        reference = evaluations["source_full_global"]
        for protocol, result in evaluations.items():
            if result.n != reference.n or result.summary["mae"] != reference.summary["mae"]:
                raise AssertionError("Calibration-only comparison changed pairs or centers")
            if not np.isclose(result.summary["coverage"] + result.summary["lower_tail"]
                              + result.summary["upper_tail"], 1, atol=1e-13):
                raise AssertionError("Overall tail fractions fail their identity")
            if protocol != reference.protocol:
                comparisons.append(paired_comparison(region, result, reference))
            if protocol in ("source_full_global", "source_last12_global", "target_last12_global", "target_rolling12_global"):
                availability.extend(cell_rows(region, result))
                availability_dates.extend(cell_date_rows(region, result))
        # Frozen Amsterdam radius; only the predictor training source changes.
        for predictor_source in PRODUCTS:
            protocol = f"predictor_{predictor_source}_frozen_amsterdam_q"
            result = evaluate(region, predictor_source, repeated_radii(region, global_full), protocol, q.cuts)
            summaries.append(result.summary)
            date_rows.extend(result.per_date)
            comparisons.append(paired_comparison(region, result, reference))
        # All selection thresholds come from source calibration histories only.
        for interval_type, base in (("global", reference), ("grouped", evaluations["source_full_grouped"])):
            base_q = global_full if interval_type == "global" else grouped_full
            for p in SELECTION_QUANTILES:
                protocol = f"selection_{interval_type}_p{p:g}"
                print(f"[{name}] {protocol}", flush=True)
                result = evaluate(region, "amsterdam", repeated_radii(region, base_q),
                                  protocol, q.cuts, threshold=float(q.thresholds[p]))
                summaries.append(result.summary)
                date_rows.extend(result.per_date)
                comparisons.append(paired_comparison(region, result, base))
                d = decomposition(base.group_n, base.group_hits, result.group_n, result.group_hits)
                composition_rows.append({"target": name, "interval_type": interval_type,
                                         "selection_quantile": p, "threshold": float(q.thresholds[p]),
                                         "all_pair_count": base.n, "retained_pair_count": result.n,
                                         "pair_retention": result.n/base.n,
                                         "coverage_all": base.summary["coverage"],
                                         "coverage_kept": result.summary["coverage"],
                                         **{key: d[key] for key in ("coverage_change", "composition_change", "within_group_change", "identity_error")}})
                for group in range(3):
                    group_rows.append({"target": name, "interval_type": interval_type,
                                       "selection_quantile": p, "roughness_group": group,
                                       "all_n": int(base.group_n[group]), "all_hits": int(base.group_hits[group]),
                                       "kept_n": int(result.group_n[group]), "kept_hits": int(result.group_hits[group]),
                                       "all_weight": d["all_weights"][group], "kept_weight": d["kept_weights"][group],
                                       "all_coverage": d["all_group_coverage"][group],
                                       "kept_coverage": d["kept_group_coverage"][group] if result.group_n[group] else None})
                cells = cell_rows(region, result, base, p)
                availability.extend(cells)
                availability_dates.extend(cell_date_rows(region, result, p))
                result.summary["n_cells_never_retained"] = sum(row["never_retained"] for row in cells)
                result.summary["n_cells_at_most_25pct_dates_with_output"] = sum(row["at_most_25pct_dates_with_output"] for row in cells)
        # Release a region's large temporary aggregates before the next region.
        del evaluations, schedules
        gc.collect()
        # Checkpoint owned derived tables; a later interrupted run can be restarted.
        write_csv(out / "summary.csv", summaries)
        write_csv(out / "per_date.csv", date_rows)
        write_csv(out / "paired_comparisons.csv", comparisons)
        write_csv(out / "coverage_autocorrelation.csv", acfs)
        write_csv(out / "selection_decomposition.csv", composition_rows)
        write_csv(out / "selection_groups.csv", group_rows)
        write_csv(out / "cell_availability.csv", availability)
        write_csv(out / "cell_date_availability.csv", availability_dates)
        write_csv(out / "calibration_usage.csv", calibration_usage)
        write_json(out / "calibration_radii.json", q.records)
    checks["real_fixed_centers_same_mae_and_pairs"] = "passed for all calibration-only protocols"
    checks["real_strict_rolling_maturity"] = "passed for all rolling windows"
    checks["real_selection_decomposition"] = "passed for all 30 selection comparisons"
    write_json(out / "results.json", {"design": {"horizon_days": 72, "history_grid_days": 12,
               "predictor": "ridge_increment", "frozen_source": "amsterdam", "nominal_coverage": .9,
               "k_values": K_VALUES, "selection_quantiles": SELECTION_QUANTILES,
               "roughness_group_cuts": q.cuts, "selection_thresholds": q.thresholds,
               "bootstrap_replicates": BOOTSTRAPS, "time_block_lengths": [6, 12],
               "ci_scope": "Separate spatial and observed-origin-block diagnostics; not a joint spatial-temporal interval",
               "safety_labels": False}, "checks": checks, "summary": summaries,
               "paired_comparisons": comparisons, "selection_decomposition": composition_rows})
    write_json(out / "synthetic_checks.json", checks)
    write_json(out / "main_resources.json", {"start_utc": start_utc,
               "end_utc": datetime.now(timezone.utc).isoformat(),
               "wall_seconds": time.monotonic()-run_start, **process_memory()})
    print(json.dumps({"complete": True, "summary_rows": len(summaries),
                      "calibration_radius_calculations": len(q.records), "output": str(out)}), flush=True)


def run_factorial(root):
    """Independent addition; do not overwrite the completed main tables."""
    started = time.monotonic()
    start_utc = datetime.now(timezone.utc).isoformat()
    out = root / "results" / "revision_calibration"
    out.mkdir(parents=True, exist_ok=True)
    regions = {name: Region.load(root, name) for name in PRODUCTS}
    q = Radii(regions)
    summaries, date_rows, comparison_rows, interactions, resources = [], [], [], [], []
    for name, region in regions.items():
        evaluated = {}
        for predictor_source in PRODUCTS:
            for calibration_source in PRODUCTS:
                cal_region = regions[calibration_source]
                item = q.get(predictor_source, calibration_source, cal_region.calibration)
                protocol = f"factorial_predictor_{predictor_source}_calibration_{calibration_source}"
                print(f"[{name}] {protocol}", flush=True)
                result = evaluate(region, predictor_source, repeated_radii(region, item), protocol, q.cuts)
                result.summary["calibration_source"] = calibration_source
                result.summary["calibration_window"] = "full static"
                result.summary["radius_mm"] = float(item[0])
                result.summary["calibration_id"] = item[1]["calibration_id"]
                result.summary["calibration_n_dates"] = item[1]["n_dates"]
                result.summary["calibration_n_pairs"] = item[1]["n_pairs"]
                for row in result.per_date:
                    row["calibration_source"] = calibration_source
                evaluated[(predictor_source, calibration_source)] = result
                summaries.append(result.summary)
                date_rows.extend(result.per_date)
                resources.append({"target": name, "predictor_source": predictor_source,
                                  "calibration_source": calibration_source,
                                  "elapsed_wall_seconds": time.monotonic()-started, **process_memory()})
        baseline = evaluated[("amsterdam", "amsterdam")]
        for p in PRODUCTS:
            rows = [evaluated[(p, c)] for c in PRODUCTS]
            if len({row.n for row in rows}) != 1 or len({row.summary["mae"] for row in rows}) != 1:
                raise AssertionError("Factorial calibration source changed predictor centers")
            for c in PRODUCTS:
                result = evaluated[(p, c)]
                comparison_rows.append(paired_comparison(region, result, baseline))
                if p == "amsterdam" or c == "amsterdam":
                    continue
                # Difference-in-differences here is only a descriptive algorithm
                # interaction contrast; it has no causal interpretation.
                combination = [evaluated[(p, c)], evaluated[(p, "amsterdam")],
                               evaluated[("amsterdam", c)], baseline]
                sign = np.asarray([1., -1., -1., 1.])
                row = {"target": name, "predictor_source": p, "calibration_source": c,
                       "reference_predictor": "amsterdam", "reference_calibration": "amsterdam",
                       "n_test_pairs": result.n}
                draw = np.asarray(region.draws)
                bootstrap_values = [estimates(e.cell_sums[draw].sum(axis=1), e.cell_count[draw].sum(axis=1))
                                    for e in combination]
                for metric_index, metric in enumerate(METRICS):
                    key = metric.replace("rmse_squared", "rmse")
                    row["interaction_"+key] = float(sum(s*e.summary[key] for s, e in zip(sign, combination)))
                    lo, hi = np.quantile(sum(s*v[:, metric_index] for s, v in zip(sign, bootstrap_values)), [.025, .975])
                    row["interaction_"+key+"_spatial_lo"] = float(lo)
                    row["interaction_"+key+"_spatial_hi"] = float(hi)
                if not np.isclose(row["interaction_mae"], 0, atol=1e-14):
                    raise AssertionError("Changing only calibration created an MAE interaction")
                interactions.append(row)
        del evaluated
        gc.collect()
    write_csv(out / "factorial_summary.csv", summaries)
    write_csv(out / "factorial_per_date.csv", date_rows)
    write_csv(out / "factorial_paired_comparisons.csv", comparison_rows)
    write_csv(out / "factorial_interactions.csv", interactions)
    write_csv(out / "factorial_resource_samples.csv", resources)
    write_json(out / "factorial_radii.json", q.records)
    write_json(out / "factorial_results.json", {"summary": summaries, "interactions": interactions,
               "radii_recomputed_for_matching_predictor_and_calibration_source": True,
               "checks": {"same_test_pairs": "passed", "calibration_only_mae_invariance": "passed",
                          "zero_mae_algorithm_interaction": "passed"}})
    write_json(out / "factorial_resources.json", {"start_utc": start_utc,
               "end_utc": datetime.now(timezone.utc).isoformat(),
               "wall_seconds": time.monotonic()-started, **process_memory()})
    print(json.dumps({"factorial_complete": True, "summary_rows": len(summaries),
                      "unique_calibration_rules": len(q.records), "wall_seconds": time.monotonic()-started}), flush=True)


def finalize_diagnostics(root):
    """Publication timing and observed date dependence, without operational claims."""
    out = root / "results" / "revision_calibration"
    regions = {name: Region.load(root, name) for name in PRODUCTS}
    announcement = np.datetime64("2026-04-29")
    timeline = []
    for name, region in regions.items():
        last_observation = np.datetime64(region.meta["dates"][-1])
        endpoint = last_observation+np.timedelta64(72, "D")
        timeline.append({"target": name, "last_observation_date_in_product": str(last_observation),
                         "last_observation_plus72_days": str(endpoint),
                         "release_announcement_date": str(announcement),
                         "announcement_lag_after_last_observation_days": int((announcement-last_observation).astype(int)),
                         "last_observation_plus72_is_before_announcement": bool(endpoint < announcement),
                         "specific_file_first_public_access_date_verified": False,
                         "simulated_test_first_origin": str(region.origin_dates[region.test[0]]),
                         "simulated_test_last_origin": str(region.origin_dates[region.test[-1]]),
                         "feature_observation_cutoff": "at or before each simulated origin",
                         "released_series_processing_window_cutoff": str(last_observation),
                         "announcement_source": "https://land.copernicus.eu/en/news/egms-annual-update-brings-new-data-improvements-to-egms-explorer"})
    write_csv(out / "release_timeline.csv", timeline)
    write_json(out / "release_timeline.json", {"milestones": timeline,
               "scope": "Version-wide announcement, not an independently archived timestamp of each file's first public access"})
    with (out / "coverage_autocorrelation.csv").open(encoding="utf-8") as handle:
        acf = list(csv.DictReader(handle))
    selected = [r for r in acf if r["protocol"] == "source_full_global"
                and int(r["observed_origin_lag"]) in (1, 6, 12)]
    with (out / "summary.csv").open(encoding="utf-8") as handle:
        summaries = list(csv.DictReader(handle))
    diagnostic = [r for r in summaries if r["protocol"] in
                  ("source_full_global", "target_rolling12_global")]
    write_json(out / "dependence_diagnostics.json", {"coverage_autocorrelation_selected_lags": selected,
               "coverage_spatial_and_time_block_intervals": diagnostic,
               "interpretation": "Observed-origin autocorrelation is descriptive; 6 and 12 are sensitivity choices, not a claim of independence. Date-block results condition on the realized calibration path, and are separate from spatial intervals."})
    # The first main run preceded resource instrumentation. Preserve its actual
    # file-time endpoints and explicitly leave its unobserved OS peak missing.
    if not (out / "main_resources.json").exists():
        start = out.stat().st_ctime
        end = (out / "results.json").stat().st_mtime
        write_json(out / "main_resources.json", {"resource_instrumentation_active": False,
                   "output_directory_creation_utc": datetime.fromtimestamp(start, timezone.utc).isoformat(),
                   "results_json_write_utc": datetime.fromtimestamp(end, timezone.utc).isoformat(),
                   "artifact_endpoint_elapsed_seconds": end-start,
                   "wall_seconds": None, "os_peak_working_set_bytes": None,
                   "scope": "Observed artifact time endpoints; do not describe this as an instrumented process runtime or RAM peak"})


def profile_main(root):
    """Measure the complete workload while preserving all scientific tables."""
    global write_json, write_csv
    original_json, original_csv = write_json, write_csv
    out = root / "results" / "revision_calibration"
    reference = json.loads((out / "results.json").read_text(encoding="utf-8"))
    def preserve_json(path, value):
        if path.name == "results.json":
            expected = {(r["target"], r["protocol"]): r for r in reference["summary"]}
            actual = {(r["target"], r["protocol"]): r for r in value["summary"]}
            if set(expected) != set(actual):
                raise AssertionError("Resource replay changed the protocol population")
            for key in expected:
                for field, before in expected[key].items():
                    if field == "elapsed_seconds":
                        continue
                    if actual[key][field] != before:
                        raise AssertionError(f"Resource replay changed {key} {field}")
        elif path.name == "main_resources.json":
            value["measurement_run"] = "Complete main computation replay with scientific table writes omitted"
            value["scientific_summary_exactly_reproduced"] = True
            value["filesystem_cache_state"] = "previously read, not a cold-start benchmark"
            original_json(out / "main_profile_resources.json", value)
    try:
        write_csv = lambda path, rows: None
        write_json = preserve_json
        run(root)
    finally:
        write_json, write_csv = original_json, original_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--factorial", action="store_true", help="Add the full 3x3 component matrix only")
    parser.add_argument("--finalize-diagnostics", action="store_true")
    parser.add_argument("--profile-main", action="store_true",
                        help="Profile the complete workload without replacing scientific tables")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    elif args.factorial:
        run_factorial(args.root.resolve())
        finalize_diagnostics(args.root.resolve())
    elif args.finalize_diagnostics:
        finalize_diagnostics(args.root.resolve())
    elif args.profile_main:
        profile_main(args.root.resolve())
    else:
        run(args.root.resolve())
        finalize_diagnostics(args.root.resolve())


if __name__ == "__main__":
    main()
