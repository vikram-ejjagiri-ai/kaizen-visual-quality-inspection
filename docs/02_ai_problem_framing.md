# 02 – AI Problem Framing

## 1. From business question to AI question
**Business question:** Is this part defective, and where?

**AI question:** Given an image of a part, how different is it from defect-free parts,
and which pixels are responsible?

- **Unit of prediction:** one image of one part
- **Output:** an anomaly score (image level) and an anomaly map (pixel level)
- **Training data:** defect-free images only

## 2. Possible solutions

| Option | How it works | Pros | Cons |
|---|---|---|---|
| A. Rule-based vision (thresholds, templates) | Classic image processing: compare to a golden template, count dark pixels | Fast, explainable | Breaks with small position or lighting changes; one rule per defect type |
| B. Supervised classification (CNN: good vs. defect) | Learn from labelled good and defective images | Strong when many labelled defects exist | **We have no defects to train on**; misses defect types never seen |
| C. Object detection / segmentation | Learn to draw boxes or masks around defects | Precise localization | Needs many pixel-labelled defects: expensive |
| D. **Anomaly detection (one-class)** | Learn only what *good* looks like; flag anything different | Needs good images only; catches unknown defects; gives heatmaps | Can flag harmless variation; needs a threshold |
| E. Reconstruction (autoencoder) | Rebuild the image; badly rebuilt areas are anomalies | Needs good images only | Often reconstructs defects too, blurry maps, needs training |

## 3. Decision
**Option D, with PatchCore** (Roth et al., CVPR 2022), because:
- it needs **only good images**, matching the data and the factory reality;
- it uses a **pretrained network without training**, so it fits in minutes on a laptop GPU;
- it produces a **pixel heatmap** for the operator (FR2);
- it is a strong, widely used method on MVTec AD (the paper reports about 99% average
  image AUROC across all categories).

**Baseline it must beat:** a global-kNN model that compares one feature vector per image with
the good images. Same backbone, but no patch-level detail. If PatchCore did not clearly beat it,
the extra complexity would not be justified.

## 4. How we measure success
- **Image AUROC:** threshold-free ranking quality (comparable with published results).
- **Pixel AUROC:** does the heatmap light up on the true defect pixels?
- **Detection rate and false reject rate** at the chosen threshold.
- **Business cost:** missed defect = 20, false reject = 1.
- **Latency per image** on GPU and CPU.

## 5. Validation strategy
MVTec AD has no defective images in training, so the threshold must be chosen on labelled test
images. To stay honest, the test set is **split in two halves, stratified by defect type**:
- **Calibration half:** choose the threshold.
- **Evaluation half:** report detection and false reject rates, used once.

Threshold-free AUROC is reported on the full test set, as in the literature.
