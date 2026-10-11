[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23299161.svg)](https://doi.org/10.5281/zenodo.23299161)

# Interval reuse across ground displacement products: analysis archive

Reproducible analysis code and executed numerical results for the manuscript
"Interval reuse across ground displacement products: coverage mismatch and local
recalibration with mature residuals" by Siyu Wang and Junjie Zhang
(corresponding author).

## Contents

| Path | Description |
|---|---|
| `analysis/revision_core.py` | Core calendar-aligned pipeline: input checks, historical grid, features, seven prediction rules, metric definitions |
| `analysis/run_revision_core.py` | Driver for the core fits and the synthetic self check |
| `analysis/run_revision_calibration.py` | Residual calibration, source and window comparisons, component controls, reference shift control, selection and decomposition |
| `analysis/verify_revision_calibration.py` | Verification of reconstructed results against packaged outputs |
| `analysis/build_revision_tables.py` | Builds the result tables from the executed outputs |
| `analysis/build_nature_figures.py` | Builds the five main figures from the executed outputs |
| `analysis/apply_sn_width_layout.py` | Auditable patch list that adapts the figure layouts to the journal text width |
| `results/` | Executed derived outputs (JSON and CSV) behind every table and figure |
| `figures/source_data/europe_locator.json` | Packaged public domain Europe locator subset used by Figure 1 |
| `DATA_MANIFEST.json`, `FILE_MANIFEST.json` | Input identifiers, sizes and SHA256 checksums |

## Data

Three ascending Calibrated EGMS products from release `2020_2024_1`:

| Product | Burst identifier | Points |
|---|---|---|
| Amsterdam | `088_0312_IW1` | 338,161 |
| Venice | `117_0266_IW3` | 64,133 |
| Jura | `088_0278_IW2` | 32,678 |

The source files are distributed by the Copernicus Land Monitoring Service at
<https://egms.land.copernicus.eu/> and are not redistributed here. Full
recomputation from raw input requires those three files with the checksums
recorded in `DATA_MANIFEST.json`.

## Environment

* Analysis: Python 3.12 with numpy, pandas and scipy (`analysis/requirements.txt`).
* Figures: matplotlib with SciencePlots and cmcrameri (`analysis/requirements_figures.txt`).

## Reproduce

```
python analysis/run_revision_core.py --synthetic-check
python analysis/run_revision_calibration.py --self-test
python analysis/verify_revision_calibration.py
python analysis/build_revision_tables.py
python analysis/build_nature_figures.py
```

The table and figure drivers read the packaged `results/` outputs, so tables and
figures can be regenerated without the raw observations. Figures are written as
vector PDF and SVG plus native 1200 dpi PNG at 130.7 mm width, the single column
text width of the journal template. Each export is checked for a minimum lettering
size of 8 pt, for text outside the canvas and for collisions between text elements
before the file is written.

## Key outputs

| File | Content |
|---|---|
| `results/core_results.json` | Prediction rule metrics, paired comparisons, interpolation and split diagnostics |
| `results/revision_calibration/summary.csv` | Coverage, width, interval score and outside counts for every interval rule |
| `results/revision_calibration/per_date.csv` | Per origin coverage and tail fractions |
| `results/revision_calibration/selection_decomposition.csv` | Exact composition and within group terms for each cutoff |
| `results/revision_calibration/cell_availability.csv` | Occupied cell retention and output availability |
| `results/revision_calibration/factorial_results.json` | Complete predictor source by residual source comparison |
