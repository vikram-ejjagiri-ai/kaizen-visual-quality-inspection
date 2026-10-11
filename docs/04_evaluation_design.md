# 04 – Evaluation Design

Code: `src/evaluation.py`, `src/pipeline.py` · Results: `reports/results.md`

## 1. Two kinds of metrics
| Metric | Question it answers | Needs a threshold? |
|---|---|---|
| Image AUROC | If I pick one defective and one good part, how often does the defective one get the higher score? | No |
| Pixel AUROC | Does the heatmap rank true defect pixels above normal pixels? | No |
| Detection rate | Share of defective parts rejected | Yes |
| False reject rate | Share of good parts wrongly rejected | Yes |
| Business cost | missed defects x 20 + false rejects x 1 | Yes |

AUROC tells us how good the model *could* be. The factory, however, only experiences the
decisions at one threshold, so both are reported.

## 2. Choosing the threshold without cheating
- The training set has **no defects**, so it cannot tell us where "defective" starts.
- Choosing the threshold on the same images we report on would make results look better
  than they will be on the line.
- Solution: split the test images into a **calibration half** and an **evaluation half**,
  stratified so every defect type appears in both.
- On the calibration half, every possible threshold is tried; the one with the **lowest business
  cost** wins (ties: the higher threshold, fewer false rejects). It is then moved halfway to the
  next lower score, so a defect scoring slightly lower than the calibration defects is still caught.
- The evaluation half is used **once** to report the numbers.

## 3. Honest limitations
- MVTec AD test sets are small (for metal_nut about 115 images), so each half has roughly 57
  images. One more or fewer caught defect changes the detection rate by several percentage
  points. The numbers are an indication, not a guarantee.
- In production, the threshold should be re-validated on the line's own camera images.
