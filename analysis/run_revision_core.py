"""Run the frozen calendar-aligned post-review core on all three EGMS products."""
from __future__ import annotations

import argparse
import ctypes
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import pandas as pd
import revision_core as core

ROOT = Path(__file__).resolve().parents[1]


def peak_working_set_bytes():
    if os.name != "nt":
        return None
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    counter = Counters(); counter.cb = ctypes.sizeof(counter)
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counter), counter.cb)
    return int(counter.PeakWorkingSetSize) if ok else None


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def global_calibration_radius(region, model):
    mask = region["phase_calibration"]
    prediction = region[f"pred_amsterdam_{model}"][:, mask].astype(np.float64)
    prediction -= region["target"][:, mask].astype(np.float64)
    return core.exact_radius(np.abs(prediction))


def baseline_results(region, radii):
    mask = region["phase_test"]
    labels, draws = region["block_labels_2000"], region["bootstrap_draws_2000"]
    result = {"region": region["meta"]["region"], "test_schedule": region["meta"]["phases"]["test"],
              "models": {}, "paired_mae_differences": []}
    point_mae = {}
    for model in core.MODELS:
        pred = region[f"pred_amsterdam_{model}"]
        sums = core.point_error_sums(pred, region["target"], mask)
        result["models"][model] = core.cluster_summary(sums, labels, draws, int(mask.sum()))
        result["models"][model]["fitting_source"] = "amsterdam" if model.startswith("ridge") else "not_fitted"
        result["models"][model]["amsterdam_global_calibration_radius_mm"] = radii[model]
        point_mae[model] = sums["mae"]
    for first, second in itertools.combinations(core.MODELS, 2):
        diff = core.cluster_summary({"mae_difference": point_mae[first] - point_mae[second]},
                                    labels, draws, int(mask.sum()))
        result["paired_mae_differences"].append({"first_model": first, "second_model": second,
                                               "difference_definition": "MAE(first) minus MAE(second), mm",
                                               **diff})
    return result


def shift_control(region, fitted, radius, shifts=(-10, -5, 0, 5, 10), chunk_points=4096):
    mask = region["phase_test"]
    points = len(region["level"])
    labels, draws = region["block_labels_2000"], region["bootstrap_draws_2000"]
    model = fitted["model"]
    b = fitted["raw_unit_coefficients"]["level"] if model == "ridge_level" else 1.
    cases = []
    for shift in shifts:
        point_sums = {key: np.empty(points, dtype=np.float64) for key in
                      ("mae", "mse", "bias", "coverage", "lower_tail", "upper_tail", "interval_score")}
        largest = 0.
        for start in range(0, points, chunk_points):
            stop = min(start + chunk_points, points)
            base = core.predict_ridge_chunk(region, fitted, start, stop)[:, mask]
            shifted = core.predict_ridge_chunk(region, fitted, start, stop, level_shift=float(shift))[:, mask]
            target = region["target"][start:stop, mask].astype(np.float64)
            error = shifted - (target + shift)
            base_error = base - target
            largest = max(largest, float(np.max(np.abs((error - base_error) - (b - 1) * shift))))
            absolute = np.abs(error)
            arrays = {"mae": absolute, "mse": np.square(error), "bias": error,
                      "coverage": absolute <= radius, "lower_tail": error > radius,
                      "upper_tail": error < -radius,
                      "interval_score": 2 * radius + 20 * np.maximum(absolute - radius, 0)}
            for key, values in arrays.items():
                point_sums[key][start:stop] = values.sum(axis=1, dtype=np.float64)
        case = core.cluster_summary(point_sums, labels, draws, int(mask.sum()))
        case.update({"constant_shift_mm": shift, "raw_unit_current_level_coefficient_b": float(b),
                     "analytic_error_change_mm": float((b - 1) * shift),
                     "max_absolute_analytic_verification_error_mm": largest,
                     "frozen_amsterdam_global_radius_mm": radius, "mean_interval_width_mm": 2 * radius})
        assert largest < 1e-9, "Reference-shift formula is not satisfied by the direct prediction implementation"
        cases.append(case)
    return {"model": model, "fixed_model_source": fitted["source"],
            "raw_unit_current_level_coefficient_b": float(b),
            "target_history_and_future_target_shifted_together": True,
            "model_coefficients_standardization_and_radius_frozen": True,
            "prediction_arithmetic": "float64 direct model; no float32 rounding in control verification",
            "algorithmic_control_not_new_geodetic_observation": True,
            "cases": cases}


def write_methods(root, result):
    def phase_table(name):
        meta = result["products"][name]
        return "\n".join(f"| {core.DISPLAY[name]} | {p} | {v['origin_count']} | {v['point_origin_pairs']:,} | "
                         f"{v['first_origin']} to {v['last_origin']} | {v['first_target']} to {v['last_target']} |"
                         for p, v in meta["phases"].items())
    interpolation = "\n".join(
        f"| {core.DISPLAY[n]} | {result['products'][n]['points']:,} | {result['products'][n]['eligible_origin_count']} | "
        f"{result['products'][n]['interpolation']['mean_interpolated_grid_fraction']:.6f} | "
        f"{result['products'][n]['interpolation']['maximum_bracket_days']} | "
        f"{result['products'][n]['dropout']['missing_exact_72_day_target_after_history_check']} |"
        for n in core.PRODUCTS)
    b = result["fitted_models"]["amsterdam"]["ridge_level"]["raw_unit_coefficients"]["level"]
    text = f"""# REV2 calendar-aligned core: executed method specification

Status: the full-point analysis actually completed. This is a post-review retrospective analysis of three previously explored ascending products, not an independent unseen holdout or a preregistered study. Calendar cuts near the original design boundaries were frozen before this run, without selecting them against REV2 test results. EGMS completed-release processing hindsight remains.

## Population and calendar eligibility

All 434,972 valid points in Amsterdam (338,161), Venice (64,133) and Jura (32,678) are retained. The original CSV SHA256 hashes were checked against the inherited provenance and agree exactly. The cached series row order and counts are unchanged. Coordinates are read as float64 directly from the same verified CSV, with unique nonmissing point IDs. No quality, velocity or outcome threshold excludes points.

Every included origin is an actual acquisition date on or after 2021-01-01, with at least 365 elapsed days since the product's first acquisition. The target must be an actual acquisition at origin+72 days. Future values are never interpolated; absent exact targets remove that origin for every model. All seven models share the same origins, points and targets. The working cache also retains eligible origins outside the evaluation phases so later calibration analyses can identify genuinely matured residuals.

The common stages are fit origins >=2021-01-01 with target <2021-05-01; calibration origins >=2021-05-01 with target <2022-10-01; test origins >=2022-11-01. The strict target inequalities prevent stage crossing; the final calibration target precedes every test origin. The source-specific realized schedules are:

| Product | Phase | Origins | Point-origin pairs | Origin dates | Target dates |
|---|---|---:|---:|---|---|
{chr(10).join(phase_table(n) for n in core.PRODUCTS)}

Only seven Amsterdam and eight Venice/Jura fitting dates remain after enforcing the exact target and calendar cutoff. The millions of fitting pairs therefore provide many spatial measurements but limited independent temporal support. Fitting-window sensitivity or independent new products would be needed to extend this empirical scope; the calendar boundaries were not expanded in response to the observed rankings.

## Historical calendar grid and missing-date accounting

The grid contains 31 values at -360,-348,...,-12,0 days relative to each origin. At an observed grid date the measurement is used exactly. Otherwise linear interpolation uses the immediately preceding and following observations, both on or before the origin. The 0-day endpoint is the actual current observation. No extrapolation is used. Requiring 365 days supplies the 360-day grid with past endpoints. Interpolation remains a transformation of processed EGMS values; it does not establish independent physical motion.

| Product | Points | Eligible origins | Fraction of grid entries interpolated | Largest past bracket, days | Origins without exact future target after history check |
|---|---:|---:|---:|---:|---:|
{interpolation}

Per-origin and per-stage interpolation fractions and largest bracket intervals, original dates, exact target indices and origin dropout counts are in each portable metadata JSON. This reporting distinguishes history interpolation from the strictly observed future target.

## Features, fixed baselines and ridge fitting

Let g_t(s) denote the historical grid value at t+s. The five unique main increment features are g_t(0)-g_t(-12), g_t(0)-g_t(-36), g_t(0)-g_t(-72), g_t(0)-g_t(-144), and r_t. Define u_j=g_t(-144+12j)-g_t(-144+12(j-1)), j=1,...,12. Then r_t=sqrt(sum((u_j-mean(u))²)/12), the population SD (ddof=0) of the twelve 12-day increments in 144 days. Values are in mm; r_t is not a rate.

The seven pre-fixed comparisons are persistence g_t(0); mean3grid mean[g_t(-24),g_t(-12),g_t(0)]; local_linear144 OLS of 13 grid values over -144,...,0 on an intercept and calendar-day time, evaluated at +72; trend_seasonal360 OLS of all 31 grid values on intercept, day time and annual sine/cosine with period 365.25 days, evaluated at +72; ridge_level with current level plus the five unique features; ridge_increment with the five features predicting the 72-day increment; and ridge_increment_duplicate with the additional feature delta144/12. The duplicate version is a marked ablation, not the main representation. The annual sine/cosine basis can use relative-origin time because both annual columns span the same phase-invariant function space as absolute time.

Amsterdam supplies the fitted main level/increment/duplicate models. The five-feature increment model is additionally fitted separately on Venice and Jura to support controlled predictor/calibrator-source comparisons. All source point-origin fitting pairs have equal weight. Means and population SDs are computed only on source fit pairs; SDs below 1e-8 are set to 1. Target centering gives an unpenalized intercept. With Z=(X-mu)/sd, coefficients solve (Z'Z/N+0.1I)beta=Z'(y-mean(y))/N. Thus the objective is mean squared residual loss plus 0.1||beta||². Lambda 0.1 is a common controlled penalty, not an optimal or test-tuned value. The raw-unit coefficients and intercept are exported.

For the level model y=d_(t+72). For increment models y=d_(t+72)-d_t and the fitted increment is added back to d_t. Constraining the level coefficient to exactly 1, exempting that fixed coefficient from the penalty, and fitting an unpenalized intercept plus the same five standardized features is algebraically identical to ridge_increment. Splitting delta144 across its original and standardized duplicate column changes its effective penalty, motivating the ablation.

## Evaluation and uncertainty

MAE, RMSE and signed prediction-minus-target bias use float64 aggregates of the common float32 prediction and target caches. All 21 model pairs report MAE(first)-MAE(second) on the same test pairs. Occupied 2 km cells are floor(easting/2000),floor(northing/2000), using original coordinates. A spatial draw samples B occupied cells with replacement and retains every selected point's test origins. Ratio-of-sums metrics preserve pair weighting. The 1,000 draws are shared across models and paired differences for each target, with NumPy linear-interpolated 2.5/97.5 percentile bounds. Fitting and calibration are held fixed. These are conditional spatial intervals for the observed period, not temporal, product-population, simultaneous or causal confidence guarantees. The full comparison grid is reported without selecting a winner or adjusting design to observed rankings.

## Reference-shift control

The frozen Amsterdam level model has raw-unit current-level coefficient b={b:.12g}. For constants c in {{-10,-5,0,5,10}} mm, all target history and target displacement values shift together, leaving increments and roughness unchanged. Direct float64 model reevaluation is compared with the analytic error change (b-1)c. The increment model has b=1 and is invariant. Coefficients, source standardization and the Amsterdam global calibration radius are frozen across c. Each radius is the exact kth absolute source-calibration residual, k=ceil((n+1)*0.9). Metrics, upper/lower target-exceedance tails and shared-block CIs are recorded for each shift. Lower target tail means target<prediction-radius; upper target tail means target>prediction+radius. This is a representation control, not a measured change in ground motion, and does not alone prove the origin of a regional failure.

## Cache interface and execution

analysis/cache/core/{{region}}/ contains metadata.json; origins/targets int64 arrays indexing the original series; phase_fit/phase_calibration/phase_test bool arrays of length M; level/target/roughness float32 arrays (N,M); features float32 (N,M,5); coords float64 (N,4) ordered easting,northing,latitude,longitude; block_labels_2000(N),block_keys_2000(B,2),bootstrap_draws_2000(1000,B); and pred_{{source}}_{{model}} float32 (N,M). Ridge_increment predictions exist for all three sources on all three target products; the other six models use the Amsterdam identifier in their cache filenames. Fixed baselines have no fitted source. Residuals are computed as pred-target, not stored in redundant arrays. load_core_region(root,name,mmap_mode='r') exposes each file stem as a dictionary key, with metadata under meta.

The run completed in {result['run']['wall_seconds']:.2f} seconds. The observed main-process peak working set was {result['run']['observed_peak_working_set_bytes']} bytes. All working arrays and outputs were on D:, and the E: original CSVs were read only. The run used Python {result['run']['python']}, NumPy {result['run']['numpy']} and pandas {result['run']['pandas']}. Cache construction, prediction and aggregation were performed in bounded point chunks. Synthetic assertions check exact future matches, boundaries, past-only interpolation, affine OLS recovery, closed-form ridge and reference-shift arithmetic; they are not included as empirical evidence.

The portable CLI can start without any .npy files. With --raw-dir it locates the exact three identifier CSVs, verifies their expected hashes, reads all date columns into a float32 D: working array and checks every required numeric field and unique point ID. Working-array fingerprints bind the raw hash, ordered dates, shape and .npy hash. An existing array without a fingerprint is compared against the original CSV before reuse. Large caches are not required in the public software archive.
"""
    (Path(root) / "CORE_METHODS.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--raw-dir", type=Path, help="Original CSV warehouse; recursively locate exact identifier filenames")
    parser.add_argument("--reuse-core-cache", action="store_true", help="Reuse completed REV2 core caches; raw hashes must match metadata")
    parser.add_argument("--synthetic-check", action="store_true", help="Only run explicitly synthetic invariants; write no empirical outputs")
    args = parser.parse_args()
    log("Synthetic invariants: " + json.dumps(core.synthetic_check()))
    if args.synthetic_check:
        return
    started = time.perf_counter()
    root = args.root.resolve()
    if root.drive.upper() != "D:":
        raise ValueError("User storage rule requires core computation and caches on D:")
    free = shutil.disk_usage(root).free
    if free < 8 * 1024 ** 3:
        raise ValueError("Insufficient D: space for the complete common-sample core cache")
    log(f"D working directory {root}; free bytes {free}")
    result = {"analysis": "post-review fixed-calendar exact72-day core", "actually_completed": False,
              "previously_explored_products": True, "independent_unseen_holdout": False,
              "processing_hindsight_remains": True, "main_unique_features": core.FEATURES,
              "duplicate_feature_model_is_ablation": True,
              "frozen_contract": {"horizon_days": core.HORIZON_DAYS, "history_availability_days": core.HISTORY_DAYS,
                                  "grid_relative_days": core.GRID_DAYS,
                                  "fit_origin_start": str(core.FIT_ORIGIN_START), "fit_target_before": str(core.FIT_TARGET_END),
                                  "calibration_origin_start": str(core.CAL_ORIGIN_START),
                                  "calibration_target_before": str(core.CAL_TARGET_END),
                                  "test_origin_start": str(core.TEST_ORIGIN_START), "ridge_penalty": core.PENALTY},
              "products": {}, "fitted_models": {}, "baselines": {}, "reference_shift_controls": {}}
    input_paths = {}
    for name in core.PRODUCTS:
        provenance = core.read_provenance(root, name)
        raw = core.resolve_raw(provenance, args.raw_dir)
        input_paths[name] = str(raw)
        meta_path = root / "analysis" / "cache" / "core" / name / "metadata.json"
        if args.reuse_core_cache and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            assert core.sha256_file(raw) == meta["raw_sha256"] == provenance["sha256"]
            assert core.sha256_file(root / "analysis" / "cache" / f"{name}_series.npy") == meta["series_cache_sha256"]
            assert meta["feature_columns"] == list(core.FEATURES)
            assert meta["dates"] == [str(d) for d in core.date_array(provenance.get("dates", [s.replace("-", "") for s in meta["dates"]]))]
            result["products"][name] = meta
            log(f"Reusing completed {name} core cache")
        else:
            log(f"Preparing {name}: verified original coordinates and calendar history")
            result["products"][name] = core.build_region_cache(root, name, args.raw_dir)
            log(f"Cache ready {name}: " + json.dumps(result["products"][name]["phases"]))
    regions = {name: core.load_core_region(root, name) for name in core.PRODUCTS}
    for source in core.PRODUCTS:
        models = core.MODELS[4:] if source == "amsterdam" else ("ridge_increment",)
        result["fitted_models"][source] = {}
        for model in models:
            log(f"Fit {source} {model}")
            fitted = core.fit_ridge(regions[source], model)
            result["fitted_models"][source][model] = fitted
            for target in core.PRODUCTS:
                core.cache_ridge_prediction(root, regions[target], fitted)
                log(f"Prediction ready {source} {model} -> {target}")
    # Reopen readers after prediction writers finish, enabling other analyses to read them safely.
    regions = {name: core.load_core_region(root, name) for name in core.PRODUCTS}
    radii = {model: global_calibration_radius(regions["amsterdam"], model) for model in core.MODELS}
    result["amsterdam_global_calibration_radii_mm"] = radii
    for name in core.PRODUCTS:
        log(f"Common-sample paired baseline evaluation {name}")
        result["baselines"][name] = baseline_results(regions[name], radii)
        result["reference_shift_controls"][name] = {}
        for model in ("ridge_level", "ridge_increment"):
            log(f"Fixed reference-shift control {name} {model}")
            result["reference_shift_controls"][name][model] = shift_control(
                regions[name], result["fitted_models"]["amsterdam"][model], radii[model])
        core.write_json(root / "results" / "core_results_in_progress.json", result)
    result["actually_completed"] = True
    result["run"] = {"wall_seconds": time.perf_counter() - started,
                     "observed_peak_working_set_bytes": peak_working_set_bytes(),
                     "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "space_free_before_bytes": free,
                     "working_drive": "D:", "raw_sources_read_only": True,
                     "command": "python analysis/run_revision_core.py",
                     "core_module_sha256": core.sha256_file(Path(core.__file__)),
                     "driver_sha256": core.sha256_file(Path(__file__))}
    core.write_json(root / "results" / "core_results.json", result)
    core.write_json(root / "results" / "core_local_run_manifest.json", {**result["run"], "raw_input_paths": input_paths})
    write_methods(root, result)
    log("Completed: " + json.dumps(result["run"]))


if __name__ == "__main__":
    main()
