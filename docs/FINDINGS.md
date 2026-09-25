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

## Scoreboard

Public leaderboard, 2026-09-25. Best is **42.14**, rank about 66 of 322.

| submission | public EMD |
|---|---|
| **reading and model blended, weight 0.45** | **42.14** |
| same, weight 0.40 | 42.59 |
| same, weight 0.50 | 42.70 |
| same, weight 0.64 | 44.59 |
| reading alone | 50.02 |
| model shape warped onto the reading | 53.45 |
| model shape shifted onto the reading | 62.65 |
| texture model alone | 68.07 |
| constant training median | 90.28 |

The blend weight is now pinned: 0.45 is the floor of a flat minimum between
0.40 and 0.50, and the curve rises either side of it.

### The warp did not survive contact

Leave-one-out said warping the model's shape onto a measured d50 and spread
should reach about 18, against 37 for the model alone. It scored 53.45, and
shifting without stretching scored 62.65. That is the sixth offline signal to
point the wrong way.

Two things inside the failure are still worth keeping. The readings taken
straight off the scale bars beat the ones "refined" using the segmentation,
53.45 against 57.11, so the segmentation's bracketing of Muenster and
Audorfring made those two numbers worse rather than better. And stretching to
the read spread beat shifting alone by 9 EMD, so the spread estimate carries
real information even though the whole is worse than the blend.

The pattern across every attempt is consistent: any single source of truth -
the model alone, the reading alone, the reading imposed on the model - loses
to averaging two of them. The blend works because its two halves are wrong in
different directions, and anything that makes one half dominate gives that up.

## Reading the photographs beats the model

| submission | public EMD |
|---|---|
| **even blend of the reading and the model** | **42.70** |
| blend at weight 0.64 | 44.59 |
| blend at 0.6, with a fines tail added | 48.04 |
| the reading alone | 50.02 |
| the reading alone, with a fines tail added | 56.04 |
| texture model alone | 68.07 |
| constant training median | 90.28 |

The texture model pools training curves, so it cannot predict a soil coarser
than any it was trained on. Two of the ten test soils are exactly that.
Muenster is rounded cobbles of 40-60 mm at 9-10 m depth; Testfeld Lidl is
crushed aggregate to 40 mm. The coarsest training soil has d50 = 6.1 mm. The
model placed both at 4.4 mm and had 86% of Muenster passing 20 mm, which on a
metric that averages absolute error across eleven diameters is worth tens of
points on those two samples alone.

Reading the diameter off the photograph has no such ceiling. Each photo was
cropped to a fixed *physical* square using the camera scale - 25 mm for sands,
120 mm for gravels - and rendered with 1, 10 and 50 mm bars drawn on it. d50
was then read against the bars. That alone scores 50.02.

The spread is not read off the photograph. A photo shows the coarse fraction
and hides fines that coat the grains, so sigma comes from the soil's character,
anchored on the 24 training curves: fitted lognormal sigma runs 0.40 for the
most uniform to 1.38 for the best graded, median 0.88.

### The blend is better than either half

An even blend scores 42.70, against 50.02 and 68.07 for its two halves. The
two disagree in different directions - the model is systematically too fine
because the iPhone photos arrive sharper than the host-downscaled training
photos, while a reading by eye misses fines - and averaging cancels part of
both.

Two attempts to improve on it failed, and both are informative:

* **Adding an explicit fines tail.** The reading gave only 4.8% passing
  0.063 mm where the training soils average 30%, which looked like an obvious
  omission. Raising it to 15.6% moved the reading from 50.02 to 56.04 and the
  blend from 42.70 to 48.04. These soils are as clean as they look; the
  training set is simply siltier than the test set.
* **Fitting the blend weight.** A quadratic through the three measured weights
  put the optimum at 0.638 and predicted 41.45. It scored 44.59. An
  absolute-error blend is not quadratic in the weight, and three points on
  three soils cannot locate a minimum.

## Measuring the grains instead of estimating them

Reading d50 by eye works, so the obvious next step is to measure it: segment
the individual stones at the photo's true scale and compute the size
distribution directly. A watershed on the distance transform, seeded at its
local maxima, separates touching grains; each region's equivalent circle
diameter divides by the scale to give millimetres.

On the test photographs it works, and it corroborates the readings:

| soil | segmented d50 of the visible fraction | read by eye |
|---|---|---|
| Testfeld Lidl WHV | 18.0 mm | 17 mm |
| Muenster | 21.7 mm | 35 mm |
| Audorfring | 9.7 mm | 4 mm |

On the training photographs it does not, and that kills it as a general
predictor. Training frames sit at 4.55 ppm, 0.22 mm per pixel, where a fine
soil has no visible grain boundaries at all: the surface segments as a few
enormous bright regions while a gravelly soil fragments into many small ones.
The measured area coarser than 2 mm therefore correlates **negatively** with
the true mass coarser than 2 mm, r = -0.63, and the same inversion holds at
6.3 mm and 20 mm. With no usable calibration set, the measurement cannot be
turned into a curve.

It is still worth something on the three coarse test soils, where it is an
independent check on a number read by eye. Testfeld agrees within 6%. For the
other two the two methods bracket the answer: Muenster's largest clasts run
out of frame so the segmentation undercuts it, and Audorfring's counts only
the visible gravel while ignoring the sand matrix that carries much of the
mass, so it overshoots. The readings for those two move to the middle of each
bracket, 30 mm and 5 mm.

For the sands the segmentation reads 7 to 32 times the true d50, which is
exactly right: below about 1.5 mm nothing is resolvable, so it measures the
rare gravel inclusions and not the sand at all.

## The curve family was the ceiling

The readings were turned into curves by assuming a lognormal. Asking how well
that family can represent a *real* soil curve at all - fitting each training
curve as closely as the family allows - shows it cannot:

| family | best achievable fit | under 10 EMD |
|---|---|---|
| lognormal, 2 parameters | 16.28 | 8 of 24 |
| median training shape, shift only | 38.00 | 0 of 24 |
| median training shape, shift and stretch | 10.11 | 14 of 24 |
| **any other training curve, shifted and stretched** | **3.33** | **24 of 24** |

Real gradations saturate at a true maximum particle size; a lognormal only
approaches 100 asymptotically, and that mismatch alone costs 16 EMD before any
prediction error. Empirical curves do not have the problem: every training
soil can be matched to within 3.33 EMD by warping another one.

(An earlier version of this table reported 40.43 for the lognormal. That came
from an under-converged optimiser and was wrong; the correct floor is 16.28.)

## Shape from the model, position from the photograph

This splits the problem along the seam where each method is strong. The
neighbour model is good at *shape* - it finds the training soils whose texture
matches, and their curves carry a realistic gradation - and bad at *position*,
because pooling training curves cannot reach past the training range. Reading
a diameter off a scale bar is the reverse.

Leave-one-out over the training set, with the true values standing in for a
reading:

| | LOO EMD |
|---|---|
| neighbour model alone | 37.08 |
| shifted onto the right d50 | 17.19 |
| shifted and stretched onto the right spread | 8.83 |
| floor, if shape selection were perfect | 3.33 |

The reading does not have to be precise for this to pay. Degrading it to 0.15
decades on d50 and 20% on spread still gives 17.79, and shifting alone stays
ahead of not shifting until the d50 error reaches about 0.5 decades, a factor
of three.

This is also why an earlier attempt at shifting failed. That version took its
d50 from a ridge regression on the same texture features, which is heavily
regularised toward the training range and so moved nothing where it mattered.
The shift is only worth making when the diameter comes from outside the model.

## The scale is verified against a ruler in the frame

One of the five Muenster photographs has a wooden ruler lying on the stones,
centimetre marks and all. That is a physical reference inside the data, and it
checks the one number the whole pipeline rests on.

Sampling rows across the ruler's tick band and taking the dominant spatial
frequency gives a period of **14.89 px per millimetre**, stable across every
row from 0.770H to 0.790H and strongly periodic there. `ppm_updated.csv` gives
13.942 px/mm for the iPhone 14. The ratio is 1.068, **+0.029 decades**, and the
ruler is resting on top of the stones rather than on the soil surface, so it
sits nearer the camera and should read slightly large.

The camera table is therefore right to within about 7%, which is a twentieth of
the uncertainty in reading a d50 off a scale bar. Scale error is ruled out as a
source of what is left. It also confirms, independently of the earlier
first-principles check, that the resolution correction applied to the training
photographs is the right one.

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
