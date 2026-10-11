"""Stage 9 - Edge deployment: export to ONNX and measure speed.

A camera on a production line inspects a part every few hundred milliseconds,
often on a small industrial PC without a big GPU. ONNX is an open model format
that runs on almost any hardware (ONNX Runtime, OpenVINO on Intel CPUs, TensorRT).
"""
import time

import numpy as np
import onnxruntime as ort
import torch

from src.config import IMAGE_SIZE, MODELS_DIR
from src.patchcore import nearest_distance


def export_onnx(patchcore, path=None):
    path = path or MODELS_DIR / "feature_extractor.onnx"
    extractor = patchcore.extractor.to("cpu").eval()
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(extractor, (dummy,), str(path), input_names=["image"],
                      output_names=["patch_features"], opset_version=18, dynamo=True)
    patchcore.extractor.to(patchcore.device)
    return path


def _time(fn, repeats):
    fn()  # warm-up (first call is always slow)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def benchmark(patchcore, onnx_path, repeats=20):
    """Median latency per image (ms) for: features + nearest-neighbour search."""
    image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    bank_cpu = patchcore.memory_bank.cpu()
    results = {}

    def full_on(device):
        extractor = patchcore.extractor.to(device)
        bank = patchcore.memory_bank.to(device)
        x = image.to(device)

        def run():
            with torch.no_grad():
                f = extractor(x)
                flat = f.permute(0, 2, 3, 1).reshape(-1, f.shape[1])
                nearest_distance(flat, bank)
        return run

    if torch.cuda.is_available():
        results["PyTorch, GPU (" + torch.cuda.get_device_name(0) + ")"] = _time(full_on("cuda"), repeats)
    results["PyTorch, CPU"] = _time(full_on("cpu"), max(3, repeats // 4))
    patchcore.extractor.to(patchcore.device)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x_np = image.numpy()

    def onnx_run():
        f = torch.from_numpy(session.run(None, {"image": x_np})[0])
        flat = f.permute(0, 2, 3, 1).reshape(-1, f.shape[1])
        nearest_distance(flat, bank_cpu)
    results["ONNX Runtime, CPU"] = _time(onnx_run, max(3, repeats // 4))

    # Check the exported model gives the same features as PyTorch
    with torch.no_grad():
        ref = patchcore.extractor.to("cpu")(image).numpy()
    patchcore.extractor.to(patchcore.device)
    onnx_out = session.run(None, {"image": x_np})[0]
    max_diff = float(np.abs(ref - onnx_out).max())
    return results, max_diff
