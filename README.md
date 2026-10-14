# 2026 KAMP: YOLO11s + ROI-Based Defect Detection

An industrial defect detection project combining color-marker removal and inpainting, YOLO11s detection, ROI verification, NMS, and coordinate refinement.

## Project Structure

| Component | Path |
|---|---|
| Source code | `src/` |
| Configuration | `configs/` |
| Data | `data/` |
| Trained models | `models/` |
| Validation and test results | `results/` |
| Runtime dependencies | `requirements.txt` |

The dataset is split into 360 training images, 68 validation images, and 72 test images.

## Installation

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Inference

```bash
# Run inference on the test and validation sets
python src/run.py test
python src/run.py validation

# Run inference on an image or directory
python src/run.py predict --input data/raw/images --output results/custom_predictions.json
```

Supported input formats are BMP, PNG, JPG, and JPEG.

## Training

```bash
python src/run.py preprocess
python src/train.py detector --device cpu
python src/train.py verifier
python src/train_shared_quality.py train
```

## Results

| Set | Score threshold | IoU | TP | FP | FN |
|---|---:|---:|---:|---:|---:|
| validation | 0.20 | 0.50 | 67 | 0 | 1 |
| validation | 0.02 | 0.50 | 68 | 15 | 0 |
| test | 0.20 | 0.50 | 79 | 1 | 3 |
| test | 0.02 | 0.50 | 80 | 21 | 2 |

Detailed prediction outputs are available in `results/test/` and `results/validation/`.
