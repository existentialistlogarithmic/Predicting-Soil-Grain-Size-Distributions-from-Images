# Competition reference

Everything here was read from Kaggle's public competition metadata on
2026-09-22. Re-check anything marked **unverified** against the live Rules and
Evaluation tabs before relying on it.

Source: <https://www.kaggle.com/competitions/soil-grain-size-from-photos>

## The task

Predict the cumulative grain size distribution — percent of the sample passing
each sieve — of a soil sample, from photographs of its surface. The curve is
reported at eleven fixed diameters spanning five decades, from clay through the
cobble boundary.

| # | Diameter (mm) | Column name | Fraction |
|---|---------------|-------------|----------|
| 0 | 0.002 | `0.002` | clay / silt boundary |
| 1 | 0.0063 | `0.0063` | fine silt |
| 2 | 0.02 | `0.02` | medium silt |
| 3 | 0.063 | `0.063` | silt / sand boundary |
| 4 | 0.2 | `0.2` | fine sand |
| 5 | 0.63 | `0.63` | medium sand |
| 6 | 2 | `2` | sand / gravel boundary |
| 7 | 6.3 | `6.3` | fine gravel |
| 8 | 20 | `20` | medium gravel |
| 9 | 63 | `63` | gravel / cobble boundary |
| 10 | 200 | `200` | cobbles (always 100) |

## Metric

Mean log-diameter-weighted Earth Mover's Distance. For one sample:

```
EMD = sum_{i=0..9} |F_i - F̂_i| · (log10(x_{i+1}) - log10(x_i))
```

The competition score is the mean over scored samples. Lower is better.

Two facts about the weights are worth internalising:

* They sum to `log10(200 / 0.002) = 5`, which is exactly why the stated range is
  `[0, 500]` — 100 percentage points of error across the whole axis.
* All ten are within 0.4% of 0.5, because the supports are very nearly a
  geometric series. So the metric is effectively `5 × MAE` over the first ten
  supports. **It is an absolute-error metric**: the score-optimal point
  prediction is the conditional *median*, and models should be trained with an
  L1 objective, not a squared one. The eleventh support never enters the sum.

## Submission format

A CSV with exactly these twelve columns, in this order:

```
sample_id,0.002,0.0063,0.02,0.063,0.2,0.63,2,6.3,20,63,200
```

Structural constraints Kaggle enforces:

* every value in `[0, 100]`;
* each row non-decreasing across the supports;
* the `200` column exactly `100`;
* one row per `sample_id` in `sample_submission.csv`.

`soilgsd validate` checks all of these. `soilgsd.curves.project_valid` enforces
them on any prediction.

## Data files

| File | Contents |
|------|----------|
| `Training_labels_updated.csv` | one row per training sample: `sample_id` plus the eleven targets |
| `sample_submission.csv` | the test `sample_id` values and the required column order |
| `ppm_updated.csv` | camera scale in pixels per millimetre |
| `Training-All_Photos_updated/` | training photographs |
| `Test_All_Photos/` | test photographs |

Samples are photographed more than once, so photos must be pooled back to
`sample_id` before modelling, and cross-validation must be grouped so that no
photograph of a validation soil is ever seen in training.

`ppm_updated.csv` is the most important file in the archive after the labels.
Grain size is a physical length; without the pixels-per-millimetre scale, the
same soil photographed by two cameras looks like two different soils.

## Timeline and rules

| Item | Value |
|------|-------|
| Host | Lukas Leibold — Geotechnical Resilience Project |
| Launched | 2026-05-18 |
| Final submission deadline | 2026-11-30 11:00 UTC |
| Team merger deadline | 2026-11-30 11:00 UTC |
| Prize | USD 500 (1 prize) |
| Max submissions per day | 5 |
| Submissions counted for the final score | 2 |
| Max team size | 5 |
| Public leaderboard | 30% of the test set; the final standing uses the other 70% |
| Data licence | CC BY 4.0 |
| Notebook-only competition | No — file submissions are allowed |

As of the metadata snapshot: 322 teams, 333 competitors, 3,419 submissions.

### Unverified

Read these on the live Rules tab before they matter:

* whether external data and pretrained weights (e.g. ImageNet backbones) are permitted;
* how the two final submissions are selected;
* any licensing obligation attached to a winning solution.
