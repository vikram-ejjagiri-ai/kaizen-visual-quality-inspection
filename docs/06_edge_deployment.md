# 06 – Edge Deployment

Code: `src/edge.py` · Results: latency table in `reports/results.md`

## Why edge?
Inspection runs next to the line, often on an industrial PC without a large GPU and sometimes
without a reliable network. Sending every image to the cloud adds delay and a single point of
failure. The model therefore has to run locally and fast.

## What the pipeline does
1. **Export** the feature extractor to **ONNX**, an open format that runs on many runtimes:
   ONNX Runtime (any CPU), OpenVINO (Intel CPUs and iGPUs, common in industrial PCs),
   TensorRT (NVIDIA Jetson and GPUs).
2. **Check correctness:** ONNX output is compared with PyTorch output; the maximum difference is
   reported (should be tiny, about 1e-4 or smaller).
3. **Benchmark** the full per-image work (features + nearest-neighbour search) on:
   PyTorch GPU (if available), PyTorch CPU, ONNX Runtime CPU. Median of repeated runs, after
   a warm-up run.

## How to read the results
- Compare the latency with the line's **cycle time** (e.g. one part every 0.5 seconds).
- The nearest-neighbour search grows with the memory bank. If CPU latency is too high:
  lower the coreset ratio (e.g. 1% instead of 10%), use a smaller image size, or use an
  approximate search library (e.g. FAISS).
- Next steps for a real deployment: OpenVINO conversion with INT8 quantization, and
  packaging the model as a small REST service in Docker next to the camera.
