# Learning Guide

Work through the stages in order: read the doc, look at the code it points to, then answer
the questions without looking. Answers are hidden below each block.

---

## Setup and first run (do this first)
Follow README steps 1-3. When `python -m src.pipeline` finishes:
1. Open `reports/results.md`: these are **your** results.
2. Open the figures in `reports/figures/`.
3. Start the review app and click through the review queue.
4. Commit and push (the `data/` folder and model files are ignored automatically).

---

## Stage 1-2: Problem and AI framing
**Read:** `docs/01_problem_definition.md`, `docs/02_ai_problem_framing.md`

1. Why can't we train a normal "good vs. defect" classifier here?
2. What is the big advantage of anomaly detection in a factory?
3. What is the main weakness of anomaly detection?

<details><summary>Answers</summary>

1. The training set contains no defects, and real factories rarely have enough labelled examples of every defect type.
2. It needs only good parts (plentiful) and can catch defect types never seen before.
3. It flags *unusual*, not *defective*: harmless variation (dust, reflections) can be flagged, so a threshold and a human check are needed.
</details>

---

## Stage 3: PatchCore
**Read:** `docs/03_method_patchcore.md` · **Code:** `src/features.py`, `src/patchcore.py`

1. Why do we use layer2 and layer3 of the network, but not the last layer?
2. What is the memory bank, and why do we keep only 10% of it?
3. Why is the coreset chosen "farthest point first" instead of randomly?
4. Why is the image score the *maximum* of the heatmap, not the average?
5. Why is the global-kNN baseline weaker on small scratches?

<details><summary>Answers</summary>

1. Middle layers describe textures and shapes at a useful scale; the last layer describes ImageNet object classes, which is too abstract for small defects.
2. All patch descriptions of good images. 10% keeps the search fast with almost no accuracy loss, because many patches are near-duplicates.
3. Random sampling mostly keeps common patches and may drop rare-but-normal ones (e.g. edges); farthest-first covers all kinds of normal.
4. One defective patch is enough to make the whole part defective; an average would dilute it.
5. A small scratch changes only a few patches, which barely moves the average over the whole image.
</details>

---

## Stage 4: Evaluation
**Read:** `docs/04_evaluation_design.md` · **Code:** `src/evaluation.py`

1. What is the difference between AUROC and detection rate?
2. Why is the test set split into calibration and evaluation halves?
3. Why is the threshold moved a little lower after it is chosen?

<details><summary>Answers</summary>

1. AUROC measures ranking quality over all thresholds; detection rate is what happens at the one threshold the factory actually uses.
2. Choosing the threshold on the same images you report on makes results look better than they will be in production.
3. A threshold placed right at the lowest-scoring calibration defect is fragile; a slightly lower one catches a new defect scoring a bit lower, which matters because missed defects are 20x more costly.
</details>

---

## Stage 5: Explainability and review
**Read:** `docs/05_explainability_and_review.md` · **Code:** `app/review_app.py`

1. Look at `missed_defects.png` and `false_rejects.png` (if they exist). What do they have in common?
2. What is the overrule rate, and why does a quality manager care about it?

<details><summary>Answers</summary>

1. Depends on your run. Typical: very small or low-contrast defects get missed; reflections or unusual but harmless texture get falsely rejected.
2. The share of AI decisions the operator overrules. Rising overrule rate means the line or camera has changed and the model needs refitting.
</details>

---

## Stage 6: Edge deployment
**Read:** `docs/06_edge_deployment.md` · **Code:** `src/edge.py`

1. Why export to ONNX?
2. Your latency results: is the CPU fast enough for a line with one part every 0.5 s? If not, what would you change?

<details><summary>Answers</summary>

1. ONNX runs on many runtimes and hardware types (ONNX Runtime, OpenVINO, TensorRT), not only PyTorch, which suits industrial PCs.
2. Check your table. Options: smaller coreset (1%), smaller image size, approximate nearest-neighbour search (FAISS), OpenVINO with INT8.
</details>

---

## Your 60-second pitch (fill in your numbers from reports/results.md)
> "I built an AI visual inspection system for industrial parts on the MVTec AD benchmark.
> Because factories have plenty of good parts but few defects, I used anomaly detection:
> PatchCore, which I implemented from scratch in PyTorch, learns only from defect-free images
> and flags any patch that looks unlike them. It reached an image AUROC of ___ versus ___ for a
> global baseline, and its heatmaps show where each defect is (pixel AUROC ___). I chose the
> decision threshold by business cost on a separate calibration set, so on unseen images it
> caught ___ of ___ defects while rejecting ___ good parts. Rejected parts go to an operator
> review screen that logs every decision, and I exported the model to ONNX and benchmarked it
> at ___ ms per image on CPU for edge deployment."
