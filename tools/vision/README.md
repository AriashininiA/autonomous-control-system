# Vision Tools

Development utilities migrated from the original vision lab.

## Runtime vs Tools

- Runtime models used by the demo live in `data/vision/models/`.
- Runtime images/calibration inputs live in `data/vision/resource/` and `data/vision/calibration/`.
- Training notebooks, conversion scripts, and small training artifacts live here under `tools/vision/`.

## Contents

- `training/f110_yolo_training.ipynb`: original training notebook. It still contains Colab/Google Drive cells, so treat it as a reproducibility/reference notebook before adapting paths for local training.
- `training/f110_yolo_architecture.py`: F1TENTH YOLO model definition used by the notebook and ONNX export script.
- `conversion/convert_onnx.py`: exports `data/vision/models/model_78.pt` to `data/vision/models/model_78.onnx` by default.
- `conversion/convert_trt.py`: optional TensorRT engine builder for Jetson/GPU environments.
- `artifacts/loss_78.npy`: saved loss curve from training.
- `artifacts/labels.npy`: labels array from the training dataset. The full image dataset is intentionally not copied into the unified portfolio project.

