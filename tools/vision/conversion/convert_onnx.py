import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAINING_DIR = PROJECT_ROOT / "tools" / "vision" / "training"
DEFAULT_PT = PROJECT_ROOT / "data" / "vision" / "models" / "model_78.pt"
DEFAULT_ONNX = PROJECT_ROOT / "data" / "vision" / "models" / "model_78.onnx"

sys.path.insert(0, str(TRAINING_DIR))

# USAGE:
# python tools/vision/conversion/convert_onnx.py

def convert(pt_path, onnx_path, height=180, width=320):
    try:
        import torch
        from f110_yolo_architecture import F110_YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("ONNX conversion requires torch. Install PyTorch before using this tool.") from exc

    pt_path = Path(pt_path).expanduser().resolve()
    onnx_path = Path(onnx_path).expanduser().resolve()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    if not pt_path.is_file():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {pt_path}")

    print("Loading model...")

    device = torch.device("cpu")

    # Initialize model
    model = F110_YOLO()
    model.load_state_dict(torch.load(str(pt_path), map_location=device))
    model.to(device)
    model.eval()

    print("Model loaded")

    # Dummy input for ONNX export
    dummy_input = torch.randn(1, 3, height, width).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        export_params=True,
    )

    print("ONNX export complete!")
    print(f"Saved to: {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", type=str, default=str(DEFAULT_PT))
    parser.add_argument("--onnx", type=str, default=str(DEFAULT_ONNX))
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--width", type=int, default=320)

    args = parser.parse_args()

    convert(args.pt, args.onnx, args.height, args.width)
