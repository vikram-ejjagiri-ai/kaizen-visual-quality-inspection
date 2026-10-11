# Portfolio update (after your first real run)

Fill the blanks from `reports/results.md`, or send the file to Claude to finalize the text.
Only publish numbers that your own run produced.

```js
  {
    id: "visual-quality-inspection",
    number: "03",
    title: "AI Visual Quality Inspection",
    shortTitle: "Visual Inspection",
    category: "Industrial AI",
    tags: ["Computer Vision", "Anomaly Detection", "Edge AI"],
    description:
      "Detects and localizes defects on industrial parts from camera images, trained on defect-free parts only, with an operator review screen and edge deployment.",
    objective:
      "Automate visual quality inspection where defects are rare and unpredictable: flag defective parts, show where the defect is, and keep a human in the loop.",
    achievements: [
      "Implemented PatchCore anomaly detection from scratch in PyTorch on the MVTec AD benchmark, using only defect-free training images.",
      "Reached an image AUROC of ___ (vs. ___ for a global-feature baseline) and a pixel AUROC of ___ for defect localization.",
      "Chose the decision threshold by business cost on a separate calibration set: caught ___ of ___ defects on unseen images while rejecting ___ good parts.",
      "Built a Streamlit operator review screen with anomaly heatmaps, confirm/overrule decisions and an audit log.",
      "Exported the model to ONNX and benchmarked it for edge deployment: ___ ms per image on GPU, ___ ms on CPU."
    ],
    tools: ["Python", "PyTorch", "ONNX Runtime", "Streamlit", "scikit-learn", "Pytest"],
    visual: "sensor",
    image: "projects/visual-quality-inspection.png",
    github: "https://github.com/vikram-ejjagiri-ai/kaizen-visual-quality-inspection"
  },
```
Card image: copy `reports/figures/heatmap_examples.png` to `portfolio\public\projects\visual-quality-inspection.png`.
