# 03 – Method: PatchCore, step by step

Code: `src/features.py`, `src/patchcore.py`

## The idea in one picture
```
GOOD training images ──► pretrained CNN ──► 1,024 patch descriptions per image
                                                   │
                                     keep a representative 10% (coreset)
                                                   ▼
                                            MEMORY BANK of "normal"

NEW image ──► pretrained CNN ──► 1,024 patch descriptions
                                       │ for each patch: distance to nearest
                                       │ normal patch in the memory bank
                                       ▼
                          32x32 grid of distances ──► upsample + smooth ──► HEATMAP
                                                           │
                                                 image score = max of heatmap
```

## 1. Feature extraction (no training)
- Backbone: **Wide-ResNet-50**, pretrained on ImageNet (1.2 million everyday photos).
- We take two intermediate layers:
  - **layer2:** small textures and edges (good for scratches)
  - **layer3:** larger shapes and parts (good for bent or missing areas)
- The very last layers are skipped: they describe "what object is this" (ImageNet classes),
  which is too abstract for tiny defects.
- Each feature is averaged with its 3x3 neighbours, so each patch also "sees" its context.
- Result for a 256x256 image: a **32x32 grid**, each cell described by **1,536 numbers**.

## 2. Memory bank and coreset
- All patches of all good images are collected (for ~220 images: ~225,000 patches).
- Searching all of them for every new patch would be slow, and many are near-duplicates.
- **Greedy coreset:** repeatedly add the patch that is farthest from all patches picked so far,
  until 10% are picked. This keeps rare-but-normal patterns (e.g. the edge of the thread),
  unlike random sampling, which would mostly keep the common ones.
- A random projection to 128 dimensions makes the selection fast without changing which
  patches are far apart (Johnson-Lindenstrauss lemma).

## 3. Scoring a new part
- Each of its 1,024 patches gets the **distance to its nearest neighbour** in the memory bank.
- Upsampled to 256x256 and smoothed (Gaussian, sigma 4): the **anomaly map**.
- **Image score = the maximum** of the map: one bad patch is enough to make a part defective.

## 4. Why not train a network?
Training needs many examples of what to detect. We have none (no defects in training) and
we want to catch defects we have never seen. PatchCore turns the problem around: describe
normal precisely, and everything else stands out.

## Baseline: global kNN
Same backbone, but the 32x32 grid is averaged into **one** 1,536-number vector per image.
Score = distance to the nearest good training image. A small scratch barely changes the
average, which is why this baseline is weaker and cannot produce a heatmap.
