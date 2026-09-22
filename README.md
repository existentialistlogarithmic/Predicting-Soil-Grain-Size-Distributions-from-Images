# Predicting Soil Grain Size Distributions from Images

A CPU-only pipeline for the Kaggle competition
[soil-grain-size-from-photos](https://www.kaggle.com/competitions/soil-grain-size-from-photos):
predict a soil sample's cumulative grain size distribution — percent passing at
eleven fixed sieve diameters from 0.002 mm to 200 mm — from photographs of its
surface.

Scored by mean **log-diameter-weighted Earth Mover's Distance**, range
`[0, 500]`, lower is better. Deadline **2026-11-30 11:00 UTC**, five
submissions per day. Full reference in [`docs/COMPETITION.md`](docs/COMPETITION.md).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

make synthetic                              # a fake dataset, no Kaggle account needed
python -m soilgsd cv     --data-root data/synthetic --models constant ridge knn gbt
python -m soilgsd submit --data-root data/synthetic --model blend:ridge+gbt
```

That runs the whole pipeline in about twenty seconds and proves the wiring
works. The synthetic soils are procedurally rendered so their texture scale
really does track their grain size — useful for catching bugs, worthless as a
score estimate.

## Getting the real data

The photos are under competition terms, so you need a Kaggle account:

1. Accept the rules on the
   [competition page](https://www.kaggle.com/competitions/soil-grain-size-from-photos/rules).
2. Create an API token (Kaggle → Settings → API → *Create New Token*) and save
   it to `~/.kaggle/kaggle.json`.
3. `make download` — or `./scripts/download_data.sh` — which unpacks into `data/raw/`.

Then:

```bash
make features      # extract and cache photo features (the slow step)
make cv            # grouped cross-validation against the competition metric
make submit        # writes artifacts/submissions/<model>.csv
make validate SUBMISSION=artifacts/submissions/gbt.csv
```

Nothing under `data/` or `artifacts/` is committed, and `kaggle.json` is
gitignored.

## How it works

### Predictions are valid by construction

A submission is rejected unless every row is non-decreasing, lies in `[0, 100]`
and ends at exactly 100. Rather than patch that up at the end, every model
returns curves through `soilgsd.curves.project_valid`, and `make_submission`
refuses to write a file that fails the validator. `soilgsd validate` runs the
same checks Kaggle does, so a broken file costs you nothing from your five
daily slots.

### Features live on the millimetre axis

Grain size is a physical length, but a pixel is not — the same soil shot from
twice the distance has half the apparent grain size. `ppm_updated.csv` gives
pixels per millimetre, so every photo is first **resampled to a fixed physical
resolution** (4 px/mm by default). After that step one pixel means 0.25 mm in
every photo, whatever camera took it.

A Laplacian pyramid on the rescaled image then becomes a *granulometry*: the
contrast energy at level `l` measures how much texture lives at a physical
scale of `0.25 × 2^l` mm. Those band energies are normalised into a profile and
summarised by their centroid on the log10-mm axis — which is directly
comparable with the axis the target itself is indexed by. Colour, chroma and
gradient statistics ride alongside so a model can separate grain *size* from
grain *contrast*.

Two tests in `tests/test_data_and_features.py` pin this down: one shoots the
same synthetic soil at two magnifications and asserts the centroid agrees to
within 0.12 decades; the other asserts a coarser soil really does push the
centroid coarser.

### The metric is absolute error, so everything is L1

The ten interval weights are all within 0.4% of 0.5, so the score is
effectively `5 × MAE` over the first ten supports. That has real consequences,
and they are baked in:

* the constant baseline is the training **median** curve, not the mean (a
  pointwise median of monotone curves is itself monotone, so it is valid);
* `GradientBoostedCurve` fits `loss="absolute_error"`;
* `NeighbourCurve` pools its neighbours by median.

### Validation is grouped

Each sample is photographed several times, and several samples can come from
one physical soil. A random split would leak a near-duplicate into validation
and flatter every model. Features are pooled per `sample_id` before modelling,
and `GroupKFold` splits on a soil key — set `group_pattern` in
`configs/default.yaml` to a regex once you can see how the host names samples.

## Models

| Name | What it does |
|------|--------------|
| `constant` | the training median curve; the number every image model must beat |
| `ridge` | ridge onto the ten free supports, `alpha` chosen by leave-one-out GCV inside `fit` |
| `knn` | median of the `k` nearest training curves — cannot produce an unphysical curve |
| `gbt` | one absolute-error gradient-boosting regressor per free support (default) |
| `blend:a+b` | convex blend of any of the above |

On the synthetic rig, the constant baseline scores 70.0 and the image models
land between 20 and 25:

```
$ python -m soilgsd make-synthetic --out data/synthetic --n-train 40 --n-test 12
$ python -m soilgsd cv --data-root data/synthetic --models constant ridge knn gbt
constant: OOF weighted EMD = 70.0224 (fold sd 20.617)
ridge:    OOF weighted EMD = 20.2745 (fold sd  4.896)
knn:      OOF weighted EMD = 24.9914 (fold sd  4.121)
gbt:      OOF weighted EMD = 24.2981 (fold sd  5.142)
```

Those numbers describe the rig, not the leaderboard. `gbt` trails `ridge` here
only because forty samples is far too few for it; expect that to reverse on the
real data.

## Layout

```
src/soilgsd/
  constants.py   the eleven supports, column names, metric weights
  metric.py      the competition metric
  curves.py      validity projection, cumulative <-> bin-mass conversions
  data.py        file discovery, loading, photo-to-sample indexing
  features.py    scale-calibrated multi-scale texture features
  models.py      curve models, all returning valid predictions
  evaluate.py    grouped cross-validation
  validate.py    the submission checker
  pipeline.py    feature caching, CV runs, submission writing
  synthetic.py   the offline test rig
  cli.py         python -m soilgsd <command>
tests/           61 tests, no Kaggle data required
docs/            competition reference
configs/         pipeline settings
```

## Where the score is likely to come from

* **The fine end is the hard part.** Clay at 0.002 mm and silt at 0.02 mm are
  orders of magnitude below what a photograph resolves. No texture feature can
  see them; they have to come from colour, gloss and the coarse fraction.
  `soilgsd cv` prints the worst supports by MAE so you can watch this directly.
* **Scale errors are fatal.** A photo whose `ppm` is wrong or missing shifts its
  whole texture profile by however many octaves the error is worth. Check the
  `[warn]` lines the pipeline prints about unmatched photos.
* **A pretrained CNN is the obvious next step** — fine-tune a small backbone on
  physically-rescaled crops with an L1 loss on the curve, predicting bin masses
  through a softmax so monotonicity is free. `requirements-vision.txt` is there
  for it. Check the Rules tab on pretrained weights first; that permission is
  listed as unverified in `docs/COMPETITION.md`.

## Tests

```bash
make test
```

61 tests covering the metric against its reference formula (including that the
weights sum to 5 and the worst case is exactly 500), curve validity, the
submission checker, every model, the scale calibration, and the CLI end to end.
None of them need the competition data.
