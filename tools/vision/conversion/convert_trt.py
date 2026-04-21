import os
import time
import argparse
from pathlib import Path

import numpy as np
import time

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ONNX = PROJECT_ROOT / "data" / "vision" / "models" / "model_78.onnx"
DEFAULT_FP32_ENGINE = PROJECT_ROOT / "outputs" / "vision" / "model_fp32.engine"
DEFAULT_FP16_ENGINE = PROJECT_ROOT / "outputs" / "vision" / "model_fp16.engine"

# USAGE:
# python3 tools/vision/conversion/convert_trt.py --benchmark


TRT_LOGGER = None


def require_tensorrt():
    try:
        import tensorrt as trt
    except ModuleNotFoundError as exc:
        raise RuntimeError("TensorRT conversion requires tensorrt.") from exc
    global TRT_LOGGER
    if TRT_LOGGER is None:
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    return trt


def require_pycuda():
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError("TensorRT benchmarking requires pycuda.") from exc
    return cuda


def build_engine(onnx_path, engine_path, use_fp16=False, workspace_mb=1024):
    trt = require_tensorrt()
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    builder = trt.Builder(TRT_LOGGER)

    try:
        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(explicit_batch)
    except Exception:
        network = builder.create_network(0)

    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("Failed to parse ONNX.")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)

    if use_fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("FP16 enabled")
        else:
            print("FP16 requested, but platform_has_fast_fp16=False")

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build serialized TensorRT engine.")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"Saved engine to: {engine_path}")
    return True


def benchmark_inference(engine_path, warmup=10, runs=100):
    trt = require_tensorrt()
    cuda = require_pycuda()
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # input: (1, 3, 180, 320), output: (1, 5, 5, 10)
    input_data = np.random.rand(1, 3, 180, 320).astype(np.float32)
    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(1 * 5 * 5 * 10 * 4)
    bindings = [int(d_input), int(d_output)]

    for _ in range(warmup):
        cuda.memcpy_htod(d_input, input_data)
        context.execute_v2(bindings)

    t0 = time.time()
    for _ in range(runs):
        cuda.memcpy_htod(d_input, input_data)
        context.execute_v2(bindings)
    avg_ms = (time.time() - t0) * 1000.0 / runs

    print(f"{engine_path}: avg inference = {avg_ms:.3f} ms")
    return avg_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default=str(DEFAULT_ONNX), help="Path to ONNX model")
    parser.add_argument("--fp32_engine", type=str, default=str(DEFAULT_FP32_ENGINE))
    parser.add_argument("--fp16_engine", type=str, default=str(DEFAULT_FP16_ENGINE))
    parser.add_argument("--workspace_mb", type=int, default=1024)
    parser.add_argument("--benchmark", action="store_true")

    args = parser.parse_args()
    Path(args.fp32_engine).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    Path(args.fp16_engine).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    print("Building FP32 engine...")
    build_engine(
        onnx_path=args.onnx,
        engine_path=args.fp32_engine,
        use_fp16=False,
        workspace_mb=args.workspace_mb,
    )

    print("Building FP16 engine...")
    build_engine(
        onnx_path=args.onnx,
        engine_path=args.fp16_engine,
        use_fp16=True,
        workspace_mb=args.workspace_mb,
    )

    if args.benchmark:
        print("\nBenchmarking inference...")
        benchmark_inference(args.fp32_engine)
        benchmark_inference(args.fp16_engine)


if __name__ == "__main__":
    main()
