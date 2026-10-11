"""Audit saved calibration results and optionally compare an independent run.

Reads only scientific outputs and core metadata; writes an audit report.
Runtime, process resources and host paths are excluded from equality claims.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import math
from pathlib import Path


def rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_close(first, second, tolerance=1e-12):
    if not math.isclose(first, second, rel_tol=0, abs_tol=tolerance):
        raise AssertionError((first, second))


def audit(root):
    out = root / "results" / "revision_calibration"
    summary = list(rows(out / "summary.csv"))
    factorial = list(rows(out / "factorial_summary.csv"))
    assert len(summary) == 90 and len(factorial) == 27
    mandatory = ("mae", "rmse", "bias_prediction_minus_target", "coverage",
                 "mean_width", "interval_score", "lower_tail", "upper_tail")
    checked_metrics = 0
    for filename in ("summary.csv", "factorial_summary.csv", "per_date.csv", "factorial_per_date.csv"):
        for row in rows(out / filename):
            for field in mandatory:
                assert math.isfinite(float(row[field])), (filename, field)
                checked_metrics += 1
            assert int(row["n_pairs"]) > 0
            check_close(sum(float(row[k]) for k in ("coverage", "lower_tail", "upper_tail")), 1)
            check_close(float(row["outside_per_1000_pairs"]), 1000*(1-float(row["coverage"])), 1e-10)
            assert int(row["n_outside_pairs"]) == int(row["n_lower_tail_pairs"])+int(row["n_upper_tail_pairs"])
    invariant = 0
    for target in ("amsterdam", "venice", "alps"):
        candidate = [row for row in summary if row["target"] == target
                     and row["predictor_source"] == "amsterdam"
                     and not row["protocol"].startswith("selection_")]
        assert candidate
        reference = candidate[0]
        for row in candidate:
            for field in ("mae", "rmse", "bias_prediction_minus_target", "n_pairs", "n_test_dates"):
                assert row[field] == reference[field], (target, row["protocol"], field)
            invariant += 1
    decomposition = list(rows(out / "selection_decomposition.csv"))
    assert len(decomposition) == 30
    errors = []
    for row in decomposition:
        diff = float(row["coverage_kept"])-float(row["coverage_all"])
        terms = float(row["composition_change"])+float(row["within_group_change"])
        errors.append(abs(diff-terms))
        check_close(diff, terms)
    radius = {record["calibration_id"]: record for record in load(out / "calibration_radii.json")}
    meta = {name: load(root / "analysis" / "cache" / "core" / name / "metadata.json")
            for name in ("amsterdam", "venice", "alps")}
    rolling = 0
    for row in rows(out / "calibration_usage.csv"):
        if not row["protocol"].startswith("target_rolling"):
            continue
        record = radius[int(row["calibration_id"])]
        origin = row["test_origin"]
        source = meta[record["calibration_source"]]
        eligible = [column for column, (date, target_date) in enumerate(zip(source["origin_dates"], source["target_dates"]))
                    if date >= "2021-05-01" and target_date < origin]
        k = int(row["protocol"].split("rolling")[1].split("_")[0])
        assert record["column_indices"] == eligible[-k:], (row, record)
        assert record["last_target"] < origin
        assert record["first_origin"] >= "2021-05-01"
        assert record["n_dates"] == k
        rolling += 1
    assert rolling == 534
    for row in rows(out / "factorial_interactions.csv"):
        check_close(float(row["interaction_mae"]), 0)
    return {"status": "PASS", "main_summary_rows": len(summary),
            "factorial_summary_rows": len(factorial), "finite_metric_values_checked": checked_metrics,
            "fixed_center_same_population_rows_checked": invariant,
            "rolling_windows_verified_as_exact_latest_k_strictly_mature_dates": rolling,
            "selection_decompositions_checked": len(decomposition),
            "largest_absolute_decomposition_identity_error": max(errors),
            "tail_count_identity": "PASS", "interval_exceedance_rate_identity": "PASS",
            "factorial_MAE_interaction_identity": "PASS"}


def compare_csv(reference, replica):
    ignored = {"elapsed_seconds"}
    count = numeric = 0
    for row_number, (first, second) in enumerate(itertools.zip_longest(rows(reference), rows(replica)), 1):
        assert first is not None and second is not None, (reference.name, "row count mismatch")
        assert set(first) == set(second), (reference.name, "field mismatch")
        for field in first:
            if field in ignored:
                continue
            count += 1
            assert first[field] == second[field], (reference.name, row_number, field, first[field], second[field])
            try:
                value = float(first[field])
            except (ValueError, TypeError):
                continue
            if math.isfinite(value):
                numeric += 1
    return {"scientific_fields_exactly_equal": count, "finite_numeric_fields_exactly_equal": numeric,
            "excluded_fields": sorted(ignored & set(first or {})), "maximum_numeric_difference": 0.0}


def compare_json(first, second, path=""):
    ignored = {"elapsed_seconds", "start_utc", "end_utc", "wall_seconds", "output", "root"}
    if isinstance(first, dict):
        assert isinstance(second, dict) and set(first) == set(second), path
        return sum(compare_json(first[key], second[key], path+"/"+key)
                   for key in first if key not in ignored)
    if isinstance(first, list):
        assert isinstance(second, list) and len(first) == len(second), path
        return sum(compare_json(a, b, path+f"/{i}") for i, (a, b) in enumerate(zip(first, second)))
    assert first == second, (path, first, second)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--compare-root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    result = {"checks": audit(root)}
    if args.compare_root:
        compare_root = args.compare_root.resolve()
        result["independent_checks"] = audit(compare_root)
        relative = Path("results") / "revision_calibration"
        exclude = {"factorial_resource_samples.csv"}
        files = sorted(path for path in (root / relative).glob("*.csv") if path.name not in exclude)
        comparison = {}
        for path in files:
            comparison[path.name] = compare_csv(path, compare_root / relative / path.name)
        json_files = ("results.json", "calibration_radii.json", "factorial_results.json", "factorial_radii.json",
                      "release_timeline.json", "dependence_diagnostics.json", "synthetic_checks.json")
        json_checked = {}
        for filename in json_files:
            json_checked[filename] = compare_json(load(root / relative / filename), load(compare_root / relative / filename))
        result["independent_reconstruction_comparison"] = {
            "status": "PASS", "comparison_type": "Exact scientific field equality; no runtime/resource/path equality claim",
            "csv_files": comparison, "json_leaf_fields_exactly_equal": json_checked,
            "total_csv_scientific_fields_checked": sum(item["scientific_fields_exactly_equal"] for item in comparison.values()),
            "total_csv_finite_numeric_fields_checked": sum(item["finite_numeric_fields_exactly_equal"] for item in comparison.values()),
            "maximum_numeric_difference": 0.0,
            "replica_main_resources": load(compare_root / relative / "main_resources.json"),
            "replica_factorial_resources": load(compare_root / relative / "factorial_resources.json")}
    output = root / "results" / "revision_calibration" / "calibration_validation.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
