# EpiAutoGP integration

This directory is the model-specific adapter between the shared forecast pipeline and [`NowcastAutoGP.jl`](https://github.com/CDCgov/NowcastAutoGP).
It is not a standalone pipeline implementation.

## Responsibility boundary

The shared `ForecastPipeline` owns:

- training-window calculation and surveillance-data loading;
- daily-to-epiweekly NSSP aggregation;
- `ForecastRun`, output directories, and common serialized artifacts;
- standard plots, credible intervals, and Hubverse output.

The EpiAutoGP adapter owns:

- validation of supported target combinations;
- selection of one training series from `ForecastRun`;
- optional reporting-delay or Hubverse nowcast trajectories;
- EpiAutoGP JSON serialization and Julia invocation.

The model input is built directly from the loaded NSSP or NHSN frame.
`combined_data.tsv` remains a shared plotting and interchange artifact; EpiAutoGP does not reread it to reconstruct data already present in memory.

## Supported configurations

  | Target | Frequency          | `ed_visit_type`               |
  | ------ | ------------------ | ----------------------------- |
  | NSSP   | daily or epiweekly | `observed`, `other`, or `pct` |
  | NHSN   | epiweekly          | `observed`                    |

For epiweekly NSSP forecasts, the shared loader retains only complete MMWR weeks.
For both frequencies, the forecast horizon starts one model time step after the final selected training observation and ends on `forecast_through`.
The Julia adapter derives the number of forecast steps from those two dates and rejects partial-step horizons.
Percentage inputs are calculated from the aggregated disease and other-visit counts.

Nowcast modes are:

- `none`: fit from the observed series only, omitting at least the three most recent available calendar days;
- `reporting-delay`: inflate the recent tail of a count series using a reporting-delay PMF;
- `hubverse`: read probabilistic NHSN trajectories from one materialized `model-output/CFA-nowcastNHSN/*.parquet` artifact.

Forecasts begin at the next model time point after the final training observation: the next calendar day for daily inputs or the next epiweek for epiweekly inputs.

## Files

- `forecast_epiautogp.py`: pipeline subclass, nowcast-source resolution, and Julia invocation.
- `nowcast.py`: fixed-data nowcast source for callers supplying precomputed trajectories.
- `prep_epiautogp_data.py`: training-series selection and JSON serialization.
- `reporting_delay_nowcast.py`: reporting-delay nowcast implementation.
- `fit_epiautogp.jl`: thin Julia adapter that calls `NowcastAutoGP` and writes standardized `samples.parquet` output.
- `Project.toml` and `Manifest.toml`: pinned Julia environment.

The JSON input is written as `{model_dir}/{model_name}_input.json`.
The Julia adapter writes `{model_dir}/samples.parquet`; the shared publication stage creates the remaining standard outputs.
