# Drift-Sense

> Given a 100× / 1 nm/px reference patch (1000×1000) and a 10× / ~10 nm/px search image (1000×1000), output the center `(x, y)` of the reference inside the search image. Tie-break: closest to `(500, 500)`.



---

## Our Solution — Three Stages

The main challenge is that DRAM contains many repeated structures. When the reference patch is searched inside the larger search image, several different DRAM cells can look very similar.

A simple template-matching method can therefore find a location that looks correct but actually belongs to another repeated cell. Our solution handles this in three steps:

**Find several possible locations → decide which one is correct → refine the final location.**

### Stage 1 — Propose: Find the Most Likely Locations

First, we use **ZNCC (Zero-Normalized Cross-Correlation)** to compare the reference patch with different parts of the search image.

We do not assume that the search image is exactly 10× larger or perfectly aligned. The webinar allows about **±20% scale** and a few degrees of rotation, so Stage 1 searches a wide grid (no retraining):

- **9 scale values:** 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0
- **7 rotation values:** −3°, −2°, −1°, 0°, +1°, +2°, +3°

This gives **63 different matching searches** (was 25 on a 9–11 / ±2° grid). Runtime is higher than the ~0.54 s/pair measured on the old grid.

For each search, ZNCC produces a score at different locations. A high score means that the reference and that part of the search image look similar.

We then:

1. Find the strongest local peaks from all 25 searches.
2. Combine the peaks into one list.
3. Remove candidates that are too close to each other.
4. Keep the **top 20 candidate locations**.

Why not simply take the highest ZNCC score?

Because DRAM is repetitive. The highest score can belong to a different but visually similar DRAM cell. Instead of making this decision immediately, we keep several strong candidates.

**Result:** up to 20 possible locations are passed to Stage 2. The correct location is among these candidates about **90% of the time**.

### Stage 2 — Rank: Decide Which Candidate Is the Real Match

At this point, we have several locations that all look similar to the reference.

This is where the **CNN verifier** is used.

For every candidate, we take:

- A **128×128 crop** around that candidate from the search image
- The corresponding reference/template information
- Three additional values:
  - ZNCC score
  - Scale ratio
  - Rotation angle

The image information is provided to the CNN as a **2-channel 128×128 input**, together with the three numerical features.

The CNN then gives each candidate a score representing how likely it is to be the correct match.

An important part of training is the use of **hard negatives**.

Instead of using completely unrelated images as negative examples, we use the other high-scoring ZNCC candidates from the **same search image**.

This is important because those candidates are exactly the difficult cases: they are usually other repeated DRAM cells that look almost identical to the correct one.

So the CNN learns the actual problem we care about:

> **Which of these visually similar DRAM cells is the one that produced the reference patch?**

The CNN is therefore **not searching the entire image**. ZNCC has already done the search. The CNN only has to choose between the most promising candidates.

### Stage 3 — Refine: Get the Final Coordinate

After the CNN ranks the candidates, we select the best one.

However, the problem specification also defines what to do when multiple locations are almost equally plausible.

If several candidates have verifier scores within **0.05 of the best score**, we use the required tie-break rule:

> Choose the candidate closest to the center of the search image, `(500, 500)`.

Finally, we improve the coordinate beyond the integer pixel location.

ZNCC initially gives us an integer peak such as:

```text
x = 844
y = 286
```

We then fit a 1D parabola around that peak on the ZNCC map. That gives a small fractional offset, so the printed result can be:

```text
844.38 285.63
```

If `verifier.pt` is missing, the pipeline falls back to ZNCC + sub-pixel and still prints `x y`. If no ZNCC peaks are found at all, it prints `500 500`.

---

## Clone & Install

**Requirements**

- **Python 3.10, 3.11, or 3.12** (not 3.13 or 3.14 — NumPy/OpenCV wheels break)
- pip · Windows or Mac · GPU optional (CUDA / Apple MPS / CPU)

If `python --version` shows **3.14**, do not use that interpreter even if 3.12 is also installed. Create the venv with 3.12 explicitly.

```bash
git clone https://github.com/Jaswanthj006/Semicon-Submission.git
cd Semicon-Submission
```

**Windows** (picks 3.12 when both 3.12 and 3.14 are installed):

```bat
py -0p
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Mac / Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

After activate, `python --version` should show 3.12.x (or 3.11 / 3.10). Then run `localize.py` with that venv.

Weights ship in `model/verifier.pt` — no training needed.

---

## Test on Your Dataset

Inference never retrains. Weights: `model/verifier.pt`.

**One pair** (prints `x y`, origin top-left of the search image):

```bash
python localize.py --reference /path/to/REF.png --search /path/to/SEARCH.png
```

**Folder of pairs** (`reference/` + `search/` with matching filenames). Same `predict()` on every pair. Prints:

`sample_id,pred_x,pred_y,confidence,inference_time_ms`

```bash
python localize.py --data "C:\path\to\split"
```

**CSV list** (when organizers share paths + sample ids). Extra columns are ignored. Needs a reference path column and a search path column (`reference_path` / `search_path`, or similar):

```bash
python localize.py --pairs-csv list.csv --output predictions.csv
```

Optional `--output predictions.csv` also writes that file for folder or CSV batch.

Python API (no CLI):

```python
from localize import predict
predict("REF.png", "SEARCH.png")
# {'sample_id', 'pred_x', 'pred_y', 'confidence', 'inference_time_ms'}
```

**Smoke test:**

```bash
python localize.py --reference dataset/reference/00000.png --search dataset/search/00000.png
# Expected: ~844.38 285.63  (GT: 844.6, 285.6)
```

RGB optical PNGs work with the same commands — loaded as grayscale internally.

---

## Generate Dataset & Test

One command builds reference + search + manifest with ground truth:

```bash
python generate_dataset.py --num-samples 100 --split test --output-dir ./output --seed 2026
```

Then score the model on it:

```bash
python train.py stage3 --data output --out model --split test
```

RGB optical bonus (does not overwrite `dataset/`):

```bash
python generate_dataset.py --num-samples 20 --split optical --output-dir ./dataset_optical --seed 99 --optical
```

---

## Results

| Split | n | pass@5 | pass@2 | pass@1 | median | time |
|---|---|---|---|---|---|---|
| **eval** | 40 | 0.875 | 0.85 | 0.625 | 0.66 px | 0.54 s |
| **test** | 250 | 0.856 | 0.844 | 0.684 | 0.64 px | 0.54 s |

Per bucket (eval): easy **1.00** · scale_rot **1.00** · hard_geometry **0.80** · hard_noise **0.70**

---

## Noise — What & Why

The search image is intentionally varied to represent different acquisition conditions. These effects are used to test whether the matcher can still locate the same DRAM structure when the image quality or appearance changes.

| Artifact | Why we add it | Citation |
|---|---|---|
| Poisson shot noise | Represents variation caused by electron-dose statistics | Joy 1995 |
| Detector Gaussian | Represents SEM electronics and detector noise | Timischl 2015 |
| Raster shear + jitter | Represents scan drift during acquisition | Jones & Nellist 2013 |
| Charging streaks | Represents local charging effects on the sample | Cazaux 2004 |
| Salt-and-pepper | Represents isolated pixel-level defects | — |
| Scale 9–11, rot ±2° | Tests small changes in effective scale and orientation | Problem statement |

Speckle, barrel distortion, and vignette are deliberately **OFF**. They are not required for the intended acquisition model and would introduce additional image distortions that are outside the main problem we are trying to solve.

The goal is to keep the underlying DRAM structure the same while changing how it appears in the search image.

Full details: `references/CITATIONS.md`

---

## Failure Mode & How We Addressed It

**What fails:** wrong DRAM period — the matcher can select a look-alike cell instead of the correct one, resulting in an error of ~100–600 px.

**Why:** at 10 nm/px, DRAM contains repeated structures that can produce very similar ZNCC scores at different locations.

**How we addressed it:**

- Stage 2 CNN trained with other ZNCC peaks as **hard negatives** (the exact aliases that cause failures)
- Spec tie-break picks the candidate nearest to (500, 500) among near-tied scores
- Reduced pass@5 failures from **~45%** (ZNCC alone) to **~12.5%** (with verifier)

Remaining misses (5 out of 40 eval): IDs `00025`, `00027`, `00029`, `00032`, `00035`. Overlays in `results/failures_eval/`.

---

## Evaluation Graph

![Precision-Recall curve](results/pr_eval.png)

Precision vs recall at different error thresholds, broken down by noise bucket.
