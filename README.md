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

## Remove Black Images

First run the cleanup script in dry-run mode. It scans `data/`, writes CSV
reports, and creates a contact sheet of detected black images without modifying
the dataset:

```bash
python src/cleanup_black_images.py
```

If the report looks correct, delete the detected images:

```bash
python src/cleanup_black_images.py --action delete --yes
```

A safer alternative is to move them out of `data/` first:

```bash
python src/cleanup_black_images.py --action quarantine --yes
```

The default detector is conservative: `mean <= 30`, `p95 <= 55`,
`black_ratio >= 0.85`, and `bright_ratio <= 0.04`. If it misses images that are
still too dark, raise the thresholds slightly, for example:

```bash
python src/cleanup_black_images.py --mean-threshold 40 --p95-threshold 75
```

## Crop Inspection Area

Create cropped BMP images from an original parent directory into a new child
directory:

```bash
python src/crop_bottle_images.py data data_cropped
```

The script scans recursively and preserves the source folder structure. Crop
rules are based on filename suffix:

| Filename suffix | Crop box | Output size |
| --- | --- | --- |
| `80_2_C.bmp` | `(0, 375)` to `(600, 725)` | `600x350` |
| `26_2_C.bmp` | `(0, 400)` to `(600, 750)` | `600x350` |
| `28_2_C.bmp` | `(0, 300)` to `(600, 650)` | `600x350` |

Preview first:

```bash
python src/crop_bottle_images.py data data_cropped --dry-run
```

Overwrite existing crops:

```bash
python src/crop_bottle_images.py data data_cropped --overwrite
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

## Run Hyperparameter Experiments

Edit `experiments.json` to define defaults and grids, then preview the expanded
training commands:

```bash
python src/run_experiments.py --config experiments.json --dry-run
```

Run all experiments:

```bash
python src/run_experiments.py --config experiments.json
```

Useful controls:

```bash
python src/run_experiments.py --config experiments.json --max-runs 2
python src/run_experiments.py --config experiments.json --start-at 4
python src/run_experiments.py --config experiments.json --skip-existing
```

Each experiment writes normal training outputs under the configured
`output_dir`, and the launcher writes comparison summaries under
`output/experiment_summaries/`.

### Experiment Config Format

`experiments.json` has three important areas:

- `defaults`: arguments used by every run unless an experiment overrides them.
- `experiments[].args`: fixed arguments for one experiment group.
- `experiments[].grid`: lists of values to sweep; every combination becomes one run.

The file also includes `_parameter_reference`, which is documentation only. The
experiment runner ignores it.

### Modifiable Training Parameters

Use snake_case names in `experiments.json`; the runner converts them to CLI
flags such as `learning_rate` -> `--learning-rate`.

| JSON key | Type / choices | Default | What it controls |
| --- | --- | --- | --- |
| `data_dir` | path | `data` | Dataset root used for auto-detecting class folders. |
| `good_dir` | path or list of paths | folders containing `BUONE` | Good bottle image folders. In JSON this may be a list, which creates repeated `--good-dir` flags. |
| `defect_dir` | path or list of paths | folders containing `SCARTI` | Defective bottle image folders. In JSON this may be a list, which creates repeated `--defect-dir` flags. |
| `output_dir` | path | `output` | Parent folder for run outputs. |
| `run_name` | string | auto-generated | Specific run folder name. Usually omit this in grids so names stay unique. |
| `seed` | integer | `42` | Random seed for split generation, sampling, and model initialization. |
| `train_ratio` | float | `0.70` | Fraction of grouped images used for training. |
| `val_ratio` | float | `0.15` | Fraction used for validation and threshold tuning. |
| `test_ratio` | float | `0.15` | Fraction reserved for final test metrics. The three split ratios must sum to `1.0`. |
| `model` | `efficientnet_b0`, `resnet18`, `mobilenet_v3_small` | `efficientnet_b0` | Neural network backbone. EfficientNet is the best default; MobileNet is faster; ResNet18 is a baseline. |
| `weights` | `imagenet`, `none` | `imagenet` | Whether to fine-tune pretrained ImageNet weights or train from random initialization. |
| `image_size` | integer pixels | `384` | Square model input size after padding/resizing. Larger can help small defects but costs memory/time. |
| `epochs` | integer | `30` | Maximum training epochs. Early stopping can stop sooner. |
| `batch_size` | integer | `16` | Images per batch. Lower this for large `image_size` or memory limits. |
| `learning_rate` | float | `0.0003` | AdamW learning rate. Good sweep: `0.0001`, `0.0003`, `0.001`. |
| `weight_decay` | float | `0.0001` | AdamW regularization to reduce overfitting. |
| `freeze_backbone_epochs` | integer | `2` | Number of initial epochs where only the classifier head trains. Use `0` for full fine-tuning immediately. |
| `early_stopping_patience` | integer | `7` | Stop after this many epochs without validation improvement. |
| `balance_strategy` | `sampler`, `loss`, `both`, `none` | `sampler` | Handles class imbalance. `sampler` oversamples defects; `loss` weights positive loss; `both` combines them. |
| `threshold_policy` | `f1`, `fixed` | `f1` | Tune final defect threshold on validation F1 or use `fixed_threshold`. |
| `fixed_threshold` | float | `0.5` | Defect probability cutoff when `threshold_policy` is `fixed`; also used during epoch logging. |
| `model_selection_metric` | `auprc`, `auroc`, `f1`, `balanced_accuracy`, `loss` | `auprc` | Validation metric for choosing `best_model.pt`. `auprc` is recommended for imbalanced defect detection. |
| `limit_images_per_class` | integer or `null` | `null` | Debug mode: keeps at most N images per class before splitting. Keep `null` for real experiments. |
| `no_augment` | boolean | `false` | When `true`, disables training augmentation. Usually keep `false`. |
| `device` | `auto`, `mps`, `cuda`, `cpu` | `auto` | Training device. `auto` prefers Apple Silicon MPS, then CUDA, then CPU. |
| `num_workers` | integer | `2` | DataLoader worker processes. Use `0` if multiprocessing gives trouble. |

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
