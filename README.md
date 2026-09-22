# Predicting Soil Grain Size Distributions from Images

A CPU-only pipeline for the Kaggle competition
[soil-grain-size-from-photos](https://www.kaggle.com/competitions/soil-grain-size-from-photos):
predict a soil sample's cumulative grain size distribution — percent passing at
eleven fixed sieve diameters from 0.002 mm to 200 mm — from photographs of its
surface.

Scored by mean **log-diameter-weighted Earth Mover's Distance**, range
`[0, 500]`, lower is better. Deadline **2026-11-30 11:00 UTC**, five
submissions per day. Full reference in [`docs/COMPETITION.md`](docs/COMPETITION.md).

**24 training samples, 10 test samples, and no camera shared between them.**
That shapes every decision here — see [`docs/FINDINGS.md`](docs/FINDINGS.md)
for the measurements, including two silent traps in the archive that cost
more than any model choice.

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

The photos are under competition terms, so you need a Kaggle account.

**1. Accept the rules** on the
[competition page](https://www.kaggle.com/competitions/soil-grain-size-from-photos/rules).
Downloads return 403 until you do, however valid your credentials are.

**2. Authenticate**, either way:

```bash
# A. environment variables - best for CI and remote containers
export KAGGLE_USERNAME=your-kaggle-username
export KAGGLE_KEY=your-api-key

# B. a token file (Kaggle -> Settings -> API -> Create New Token)
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

Both the key and the token file are secrets. `kaggle.json` is gitignored; never
paste either into a chat, a commit, or an issue.

**3. Download**: `make download`, or `./scripts/download_data.sh`, which unpacks
into `data/raw/`.

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
| `knn` | median of the `k` nearest training curves, per-domain standardised (default) |
| `ridge` | ridge onto the ten free supports, `alpha` by leave-one-out GCV inside `fit` |
| `gbt` | one absolute-error gradient-boosting regressor per free support |
| `blend:a+b` | convex blend of any of the above |

Leave-one-out over the 24 training samples:

```
$ python -m soilgsd cv --models constant knn ridge gbt
constant: OOF weighted EMD = 97.2483
knn:      OOF weighted EMD = 37.0786   <- default
gbt:      OOF weighted EMD = 42.2681
ridge:    OOF weighted EMD = 42.9972
```

Fold standard deviation is around 30 on 24 samples, so read the ordering, not
the decimals. **The leave-one-out number is not the one to trust**: held out by
*camera* rather than by sample, ridge scores 89.8–99.7 — worse than predicting
the median curve — while leave-one-out flattered it at 43.0. `knn` is the
default because it is the only model whose accuracy survived a change of
camera.

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

* **Nothing may encode the camera.** Training photos are Motorola and Samsung;
  every test photo is an iPhone. Colour statistics encode white balance and
  exposure, and `meta_*` records pixels per millimetre outright — on the test
  set it takes values (13.9–19.5) far outside anything in training (~4.55).
  The default `texture` feature set drops both.
* **The fine end is unreachable.** Clay at 0.002 mm and silt at 0.02 mm sit far
  below the 4.55 ppm ceiling the training photos impose — roughly 0.22 mm per
  pixel. They are the worst supports by mean absolute error, and the only
  features that could see them are the colour features that do not transfer.
  This is the structural ceiling on the score.
* **Validate by camera, not by sample.** Leave-one-out cannot see the shift
  that decides the leaderboard. Holding out Samsung entirely is the closest
  available proxy.

## Tests

```bash
make test
```

61 tests covering the metric against its reference formula (including that the
weights sum to 5 and the worst case is exactly 500), curve validity, the
submission checker, every model, the scale calibration, and the CLI end to end.
None of them need the competition data.
