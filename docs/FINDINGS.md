# What the data actually looks like

Notes from working with the real competition archive. Everything here was
measured, not assumed; the numbers are reproducible with the commands shown.

## The dataset is tiny and the split is adversarial

| | train | test |
|---|---|---|
| samples | **24** | **10** |
| photos | 127 | 35 |
| cameras | Motorola Edge, Samsung A52, Motorola Edge 60 Fusion | iPhone 14, iPhone 16 |
| sample ids | borehole codes (`F827`, `H030`) | site names (`HPC_Audorfring`) |

Twenty-four training samples rules out anything data-hungry. It also makes
leave-one-out the only sensible validation, which is why `n_splits` defaults
to 24.

The important part is the last two rows: **no camera appears on both sides of
the split**, and the naming schemes differ completely. Any quantity that
encodes the phone rather than the soil will look informative in
cross-validation and then fail on the leaderboard.

## Trap 1: the quoted pixels-per-millimetre is wrong for most training photos

`ppm_updated.csv` gives a scale *for a stated reference resolution*. Most
training photos were not delivered at that resolution:

| camera | reference | delivered | quoted ppm | **true ppm** | photos |
|---|---|---|---|---|---|
| Motorola Edge | 4000 px | 1600 px | 11.492 | **4.597** | 69 |
| Samsung A52 | 9248 px | 1599 px | 26.330 | **4.553** | 55 |
| Motorola Edge 60 Fusion | 4096 px | 4096 px | 12.465 | 12.465 | 3 |
| iPhone 14 | 4032 px | 4032 px | 13.942 | 13.942 | 14 |
| iPhone 16 | 5712 px | 5712 px | 19.525 | 19.525 | 21 |

124 of 127 training photos are affected; **no test photo is**. Taking the
quoted number at face value reads training texture 1.3 to 2.5 octaves finer
than it really is while reading the test set correctly — a silent train/test
mismatch that no amount of model tuning recovers from.

The fix is one ratio, in `build_photo_index`:

```
true_ppm = quoted_ppm * (delivered_long_side / reference_long_side)
```

After it, every photo in the competition lands within 4.55–19.5 ppm of its
true scale, and `target_ppm: 4.0` sits just under the coarsest training
camera — which is the real resolution ceiling of this dataset.

## Trap 2: filenames and sample ids are spelled differently

`sample_submission.csv` says `HPC_Muenster_BS6_9_0-10m`. The photos are named
`iPhone14_HPC_M#U00fcnster_BS6_9,0-10m (3).JPG` — a mangled `ü`, and a comma
where the id has an underscore. `iPhone_16` also appears once among twenty
`iPhone16`. Folding both sides through `normalise_key` (decode `#U00xx`,
expand umlauts, drop punctuation, lowercase) matches **127/127 training and
35/35 test photos** with nothing left over.

## What survives a change of camera

21 of the 24 training samples were photographed by *both* a Motorola and a
Samsung, which is a natural experiment for the train→test shift. Training on
one camera and predicting the other, leave-one-out, k-nearest-neighbour:

| feature set | same camera | cross camera | cost |
|---|---|---|---|
| everything (117) | 44.6 | 73.0 | +28.4 |
| colour only (64) | 53.8 | 85.7 | +31.9 |
| **texture only (44)** | **42.7** | **39.2** | **−3.5** |

Constant-median baseline on the same 21 samples: **83.2**.

Colour encodes white balance and exposure, so it does not transfer. Ridge
regression is worse still: cross-camera it scores 89.8 / 99.7, *worse than
predicting the median curve*, while leave-one-out flattered it at 44.7. That
gap is the whole danger of this competition in one number.

## Two residual leaks the camera experiment could not see

Motorola and Samsung were both delivered at ~0.87 of the scale the pipeline
resamples to. iPhone photos are downsampled ×0.20–0.29 to reach the same
scale. So the moto↔samsung experiment cannot detect anything caused by
resampling, and two problems slipped through it:

**`__pstd` means different things on each side.** It is the spread across the
photos of one sample. A training sample is usually shot by two different
phones, so its spread is dominated by camera disagreement; every test sample
comes from a single phone. It was the largest single contributor to the
train/test feature offset, so the default `texture` set drops it.

**The feature clouds are offset.** Even on texture alone, every test sample's
nearest training neighbours were the same handful of points, and the model
predicted nearly the same curve for all ten (between-sample spread 1.98).
Standardising each domain by its own statistics fixes it:

| configuration | cross camera | test→train distance | prediction spread |
|---|---|---|---|
| texture, k=5 | 39.2 | 1.34× | 1.98 |
| texture, k=5, per-domain | 44.1 | 0.97× | 13.3 |
| no `__pstd`, k=7 | 42.2 | 1.25× | 14.5 |
| **no `__pstd`, k=7, per-domain** | **40.5** | **0.83×** | **15.0** |

Distance is to the five nearest training neighbours, as a multiple of the
largest such distance within training; below 1.0 means the test set sits
inside the training cloud. That last row is the shipped default.

## Where it stands

```
$ python -m soilgsd cv --models constant knn ridge gbt
constant: 97.25    knn: 37.08    gbt: 42.27    ridge: 43.00
```

Leave-one-out over 24 samples, so a single hard soil moves it by several
points — fold standard deviation is ~30. Treat the ordering as meaningful and
the exact value as noisy. The cross-camera number (40.5) is the more honest
estimate of leaderboard behaviour, and even that cannot see the iPhone
resampling.

## What I would try next

* **A camera-aware validation split.** Holding out *Samsung* entirely, rather
  than leave-one-out, is the closest available proxy for the real split, and
  nothing should be selected on leave-one-out alone again.
* **Matched degradation.** Downscale the iPhone photos through the same chain
  the host used on the training photos before extracting features, instead of
  correcting for it afterwards.
* **The fine end is unreachable.** Clay (0.002 mm) and silt (0.02 mm) sit far
  below 4.55 ppm — about 0.22 mm per pixel. They are currently the worst
  supports by mean absolute error and no texture feature can see them; they
  have to come from colour, which does not transfer between cameras. This is
  the structural ceiling on the score.
* **Pretrained backbones** remain unverified against the rules. With 24
  samples, a frozen feature extractor plus this same neighbour model is the
  only sane way to use one.
