# Glass Bottle Quality Control

Binary classifier for detecting defective glass bottles from frontal BMP images.

The training script auto-detects:

- good bottles: folders under `data/` containing `BUONE`
- defective bottles: folders under `data/` containing `SCARTI`

For this repository that means the defective class comes from
`data/FOTO FRONTALI SCARTI 2`.

## Setup

Create and activate a virtual environment:

```bash
/usr/bin/python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Apple Silicon, PyTorch uses the Mac GPU through the `mps` backend. The script
selects it automatically when available.

Check it with:

```bash
python -c "import torch; print('built=', torch.backends.mps.is_built()); print('available=', torch.backends.mps.is_available())"
```

If `built=True` but `available=False`, PyTorch has MPS support but the current
session is not exposing the GPU. Run from a normal local Terminal session and
use `--device mps` if you want the script to fail fast instead of falling back
to CPU.

## Train

Default training uses EfficientNet-B0 fine tuning with ImageNet weights:

```bash
python src/main.py train
```

Useful options:

```bash
python src/main.py train \
  --epochs 30 \
  --batch-size 16 \
  --image-size 384 \
  --model efficientnet_b0 \
  --weights imagenet
```

If you want to train only on the view/camera-2 good folders against the camera-2
defective folder:

```bash
python src/main.py train \
  --good-dir "data/FOTO FRONTALI BUONE 2" \
  --good-dir "data/FOTO FRONTALI BUONE 2a" \
  --defect-dir "data/FOTO FRONTALI SCARTI 2"
```

## Outputs

Each run writes a folder inside `output/` containing:

- `best_model.pt`: best checkpoint, including the tuned decision threshold
- `split.csv`: group-aware train/validation/test split
- `history.csv`: epoch metrics
- `test_metrics.json`: final test metrics
- `test_predictions.csv`: image-level predictions
- `classification_report.txt`
- `dataset_distribution.png`
- `source_distribution.png`
- `augmentation_examples.png`
- `training_curves.png`
- `confusion_matrix.png`
- `roc_curve.png`
- `precision_recall_curve.png`
- `probability_histogram.png`
- `sample_predictions.png`

## Evaluate A Checkpoint

```bash
python src/main.py evaluate --checkpoint output/<run>/best_model.pt
```

## Predict New Images

```bash
python src/main.py predict \
  --checkpoint output/<run>/best_model.pt \
  path/to/image_1.bmp path/to/image_2.bmp
```

The printed probability is the predicted probability of a defective bottle.
