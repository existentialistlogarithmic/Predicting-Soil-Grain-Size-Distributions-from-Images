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

Four submissions, 2026-09-22:

| submission | public EMD |
|---|---|
| **knn, texture, k=7, training-referenced scaling** | **68.07** |
| d50 scalar bottleneck | 85.26 |
| knn, same but per-domain standardised | 85.06 |
| constant training-median curve, no images | 90.28 |

The best model is 22 EMD better than predicting the median, and the single
change that produced most of that was **deleting a domain-adaptation step**.

### Standardising to the test batch cost 17 EMD

The first model rescaled each batch of features by that batch's own mean and
standard deviation, on the reasoning that the training and test clouds were
offset. They are: test samples sat 1.25x the training spread from their
nearest training neighbours, and the rescaling cut that to 0.83x while
visibly restoring prediction variety. Every offline diagnostic said it was
working. It scored 85.06; removing it scored 68.07.

The features are already calibrated to millimetres through the camera scale.
Rescaling them to a batch's own spread throws that calibration away, and the
calibration was the whole point. The offset between the clouds was the price
of keeping a physically meaningful axis, not a fault to be corrected.

### Ordering is not calibration

The scalar bottleneck predicts one number, log10(d50), and then pools the
training curves nearest it. It ranks the ten test soils in exactly the order
the photographs show - cobbles at Muenster coarsest, coarse gravel at Testfeld
Lidl next, gravelly sand at Audorfring third, the seven fine sands below them
(Spearman +1.00 against a visual reading, versus +0.82 for the model that
scores 68.07). It scored 85.26.

Getting the order right is not the task. Forcing the image through one scalar
collapsed ten soils onto four distinct curves, and the metric measures the
area between curves at eleven diameters, not their ranking. A check on
ordering alone cannot see that, which makes it a weak proxy in the same way
the cross-camera experiment was.

### Things that were tried and did not help

* **Cropping to the soil.** The soil sits in a tray whose dark rim is visible
  in most photos, and sampling the rim as soil is clearly wrong. Removing it
  made cross-camera EMD worse, 40.70 to 44.18, because the rim is visible to
  different extents on different cameras and cropping to it changes the field
  of view inconsistently between them. Available as `crop_to_soil`, off.
* **Matched degradation.** Training photos were downscaled hard by the host
  and test photos were not, so pushing the test photos through a similar chain
  should have closed some of the gap. Across intermediate sizes from 1600 to
  900 px and JPEG quality down to 80, the distance from test to training moved
  from 1.247 to 1.242. Resampling artefacts are not what separates the two
  sets.
* **Shrinking toward the median.** Cross-camera cost rises monotonically as
  weight leaves the model, and an absolute-error metric bounds any blend at
  the weighted average of its endpoints. Use the better predictor alone.

### On the leaderboard itself

The public split is 30% of ten samples, so **the public score is three soils**.
A few points is noise; the 22-point gap to the baseline and the 17-point gap
between the two scaling choices are not. The final standing uses the other
seven.

## The leaderboard is not what it looks like

The top of the public leaderboard sits near 0.92. That is a mean absolute
error of about 0.2 percentage points per support, across every sample. No
model predicts a sieve curve from a photograph that accurately.

The host has said so himself, in the "possible Private Test-Set Leak?" thread:

> I have also reviewed the private leaderboard, and those scores are within a
> realistic range given the training data and task difficulty. My assumption is
> therefore that the top public score reflects overfitting to the public test
> split, whose scores are visible and can be probed through repeated
> submissions. We are looking into how to address this.

And, separately: **"The public leaderboard always uses the same 3 soils out of
the 10 test soils."** Three fixed soils, eleven supports each, thirty-three
numbers, five submissions a day since May. That is enough to fit the public
split by hand without ever looking at an image, and it earns nothing on the
private seven that decide the competition.

So the public ranking should not be chased. Probing it is not attempted here:
it is the behaviour the host is actively investigating, it produces a model
that knows nothing, and it cannot transfer to the private split by
construction. A legitimately earned 68 may well finish ahead of a probed 0.92.

## Confirmed from the host's own answers

* **"The distance from the soil to the camera is always 21 cm."** That makes
  the scale computable from first principles: for a pinhole at fixed distance,
  `ppm = width_px * f35 / (210 * 36)`. Every test photo carries its
  35 mm-equivalent focal length in EXIF (all 26 mm, one lens, no zoom
  variation), giving 13.87 ppm for the iPhone 14 and 19.64 for the iPhone 16
  against 13.942 and 19.525 in `ppm_updated.csv` - agreement within 0.6%.
  The camera table is right, and the resolution correction applied to the
  training photos is what makes it usable.
* **Test photos carry GPS and timestamps; training photos carry almost no
  EXIF at all.** The locations are real German sites photographed in October
  2024. Nothing in the pipeline uses this, since turning a coordinate into a
  sieve curve would require external geotechnical records.
* **H374 was removed and re-added with a corrected label**, and the corrected
  version is fine: it ranks 19th of 24 by leave-one-out error despite being
  the only Motorola Edge 60 Fusion sample. H031 is already gone from the file.

## A fourth proxy, also wrong

H037 and H038 are by far the hardest training samples (leave-one-out errors of
103 and 136), and their photographs look finer than labels claiming more than
half the mass coarser than 2 mm. Dropping them improved leave-one-out from
37.08 to 28.21 *and* improved the independent ordering check on the test set,
from +0.818 to +0.845. Two signals agreed, one of them measured on data the
change could not have influenced.

It scored 71.67 against 68.07. On three public soils that is inside the noise,
so it is not proof the samples are fine - but it is one more reminder that on
a dataset this small, every offline signal available is weaker than it looks.
Four have now pointed the wrong way: leave-one-out on ridge, the cross-camera
experiment, a perfect visual ordering, and this.

## What I would try next

* **Stop trusting single-number proxies.** Three different ones have now been
  wrong in three different ways: leave-one-out flattered ridge by 46, the
  cross-camera experiment predicted a 43-point win that came in at 5, and a
  perfect ordering against the photographs scored 17 worse than an imperfect
  one. Any future selection should look at ordering, calibration and spread
  together, and should expect to be wrong.
* **Per-photo scale from the tray.** The tray is a fixed object visible in
  most photos, so its width in pixels gives a per-photo scale that does not
  depend on the camera table at all. A first attempt at measuring it implied
  widths from 19 to 329 mm, which says the detector is too crude rather than
  that the idea is wrong. This is the most promising untried lever, because it
  would replace the one input the whole pipeline rests on with something
  measured per photograph.
* **The fine end is unreachable.** Clay (0.002 mm) and silt (0.02 mm) sit far
  below 4.55 ppm — about 0.22 mm per pixel. They are currently the worst
  supports by mean absolute error and no texture feature can see them; they
  have to come from colour, which does not transfer between cameras. This is
  the structural ceiling on the score.
* **Pretrained backbones** remain unverified against the rules. With 24
  samples, a frozen feature extractor plus this same neighbour model is the
  only sane way to use one.
