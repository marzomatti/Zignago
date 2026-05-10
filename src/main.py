#!/usr/bin/env python3
"""Glass bottle defect classification with PyTorch.

This script is intentionally self-contained so the project can be run from a
fresh checkout:

    python src/main.py train
    python src/main.py evaluate --checkpoint output/<run>/best_model.pt
    python src/main.py predict --checkpoint output/<run>/best_model.pt image.bmp
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "zignago_matplotlib")
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageStat
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet18,
)
from tqdm.auto import tqdm


IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
CLASS_TO_IDX = {"good": 0, "defective": 1}
IDX_TO_CLASS = {value: key for key, value in CLASS_TO_IDX.items()}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_GROUP_VIEW_SUFFIXES = ("26_2_C.bmp", "28_2_C.bmp", "80_2_C.bmp")


@dataclass
class RunConfig:
    data_dir: str
    output_dir: str
    good_dirs: List[str]
    defect_dirs: List[str]
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int
    model: str
    weights: str
    image_size: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    freeze_backbone_epochs: int
    early_stopping_patience: int
    balance_strategy: str
    threshold_policy: str
    fixed_threshold: float
    device: str
    num_workers: int
    augment: bool
    model_selection_metric: str
    limit_images_per_class: Optional[int]
    sample_level: str
    group_view_suffixes: str
    group_aggregation: str


class PadToSquare:
    """Pad a PIL image to a square while preserving the original aspect ratio."""

    def __init__(self, fill: str = "median") -> None:
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width == height:
            return image

        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2

        if self.fill == "median":
            medians = ImageStat.Stat(image).median
            if image.mode == "RGB":
                fill = tuple(int(value) for value in medians[:3])
            else:
                fill = int(medians[0])
        else:
            fill = 0

        canvas = Image.new(image.mode, (side, side), fill)
        canvas.paste(image, (left, top))
        return canvas


class GlassBottleDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: transforms.Compose) -> None:
        self.paths = [Path(path) for path in frame["path"].tolist()]
        self.labels = frame["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return image, label, str(path)


class GroupedBottleDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: transforms.Compose) -> None:
        self.path_groups = [
            [Path(path) for path in json.loads(paths_json)]
            for paths_json in frame["paths_json"].tolist()
        ]
        self.labels = frame["label"].astype(int).tolist()
        self.identifiers = frame["paths_json"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.path_groups)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        images = []
        for path in self.path_groups[index]:
            with Image.open(path) as image:
                image = image.convert("RGB")
                images.append(self.transform(image))

        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return torch.stack(images), label, self.identifiers[index]


class MultiViewClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_views: int,
        aggregation: str,
        classifier: nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_views = num_views
        self.aggregation = aggregation
        self.classifier = classifier

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError(
                "Grouped bottle model expects input with shape "
                "[batch, views, channels, height, width]."
            )

        batch_size, num_views, channels, height, width = images.shape
        if num_views != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {num_views}.")

        flat_images = images.reshape(batch_size * num_views, channels, height, width)
        features = self.backbone(flat_images)
        features = features.reshape(batch_size, num_views, self.feature_dim)

        if self.aggregation == "concat":
            features = features.reshape(batch_size, num_views * self.feature_dim)
        elif self.aggregation == "mean":
            features = features.mean(dim=1)
        else:
            raise ValueError(f"Unknown group aggregation: {self.aggregation}")

        return self.classifier(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a binary glass bottle defect classifier."
    )
    subparsers = parser.add_subparsers(dest="command")

    train = subparsers.add_parser("train", help="Train, validate, and test a model.")
    add_data_args(train)
    add_model_args(train)
    train.add_argument("--output-dir", default="output")
    train.add_argument("--run-name", default=None)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--train-ratio", type=float, default=0.70)
    train.add_argument("--val-ratio", type=float, default=0.15)
    train.add_argument("--test-ratio", type=float, default=0.15)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--freeze-backbone-epochs", type=int, default=2)
    train.add_argument("--early-stopping-patience", type=int, default=7)
    train.add_argument(
        "--balance-strategy",
        choices=["sampler", "loss", "both", "none"],
        default="sampler",
        help="How to handle the strong good/defective class imbalance.",
    )
    train.add_argument(
        "--threshold-policy",
        choices=["f1", "fixed"],
        default="f1",
        help="Tune the decision threshold on validation F1, or keep a fixed threshold.",
    )
    train.add_argument("--fixed-threshold", type=float, default=0.5)
    train.add_argument(
        "--model-selection-metric",
        choices=["auprc", "auroc", "f1", "balanced_accuracy", "loss"],
        default="auprc",
    )
    train.add_argument(
        "--limit-images-per-class",
        type=int,
        default=None,
        help="Optional debug mode. Keeps at most N images per class before splitting.",
    )
    train.add_argument(
        "--sample-level",
        choices=["group", "image"],
        default="group",
        help=(
            "Use one training sample per bottle prefix ('group') or one sample "
            "per image ('image')."
        ),
    )
    train.add_argument(
        "--group-view-suffixes",
        default=",".join(DEFAULT_GROUP_VIEW_SUFFIXES),
        help=(
            "Comma-separated filename suffixes that make up a grouped bottle "
            "sample, in view order."
        ),
    )
    train.add_argument(
        "--group-aggregation",
        choices=["concat", "mean"],
        default="concat",
        help="How to combine shared-backbone features from grouped bottle views.",
    )
    train.add_argument("--no-augment", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a saved checkpoint.")
    add_model_args(evaluate, include_training_defaults=False)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split-csv", default=None)
    evaluate.add_argument("--data-dir", default="data")
    evaluate.add_argument("--output-dir", default=None)
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--num-workers", type=int, default=2)
    evaluate.add_argument(
        "--split", choices=["train", "val", "test", "all"], default="test"
    )
    evaluate.add_argument("--threshold", type=float, default=None)

    predict = subparsers.add_parser("predict", help="Predict one or more images.")
    add_model_args(predict, include_training_defaults=False)
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("images", nargs="+")
    predict.add_argument("--output-csv", default=None)
    predict.add_argument("--threshold", type=float, default=None)

    if len(sys.argv) == 1:
        return parser.parse_args(["train"])
    return parser.parse_args()


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--good-dir",
        action="append",
        default=None,
        help=(
            "Good bottle image directory. Can be repeated. If omitted, folders "
            "containing 'BUONE' under --data-dir are used."
        ),
    )
    parser.add_argument(
        "--defect-dir",
        action="append",
        default=None,
        help=(
            "Defective bottle image directory. Can be repeated. If omitted, folders "
            "containing 'SCARTI' under --data-dir are used."
        ),
    )


def add_model_args(
    parser: argparse.ArgumentParser, include_training_defaults: bool = True
) -> None:
    parser.add_argument(
        "--model",
        choices=["efficientnet_b0", "resnet18", "mobilenet_v3_small"],
        default="efficientnet_b0",
    )
    parser.add_argument(
        "--weights",
        choices=["imagenet", "none"],
        default="imagenet",
        help="Use ImageNet weights for fine tuning, or train from random init.",
    )
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help="Use 'auto' to prefer Apple Silicon MPS, then CUDA, then CPU.",
    )
    if include_training_defaults:
        parser.add_argument("--num-workers", type=int, default=2)


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "predict":
        predict_command(args)
    else:
        raise SystemExit("Choose one of: train, evaluate, predict")


def train_command(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = choose_device(args.device)
    run_dir = make_run_dir(Path(args.output_dir), args.run_name, args.model)

    good_dirs, defect_dirs = resolve_class_dirs(
        Path(args.data_dir), args.good_dir, args.defect_dir
    )
    config = RunConfig(
        data_dir=str(Path(args.data_dir)),
        output_dir=str(run_dir),
        good_dirs=[str(path) for path in good_dirs],
        defect_dirs=[str(path) for path in defect_dirs],
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        model=args.model,
        weights=args.weights,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        early_stopping_patience=args.early_stopping_patience,
        balance_strategy=args.balance_strategy,
        threshold_policy=args.threshold_policy,
        fixed_threshold=args.fixed_threshold,
        device=str(device),
        num_workers=args.num_workers,
        augment=not args.no_augment,
        model_selection_metric=args.model_selection_metric,
        limit_images_per_class=args.limit_images_per_class,
        sample_level=args.sample_level,
        group_view_suffixes=args.group_view_suffixes,
        group_aggregation=args.group_aggregation,
    )
    write_json(run_dir / "config.json", asdict(config))

    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    print("Good folders:")
    for directory in good_dirs:
        print(f"  - {directory}")
    print("Defective folders:")
    for directory in defect_dirs:
        print(f"  - {directory}")

    image_frame = scan_dataset(good_dirs, defect_dirs)
    view_suffixes = parse_group_view_suffixes(args.group_view_suffixes)
    if args.sample_level == "group":
        frame = make_bottle_group_frame(image_frame, view_suffixes)
    else:
        frame = image_frame

    if args.limit_images_per_class is not None:
        frame = limit_per_class(frame, args.limit_images_per_class, args.seed)

    frame = make_group_split(
        frame, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )
    split_csv = run_dir / "split.csv"
    frame.to_csv(split_csv, index=False)
    write_dataset_summary(frame, run_dir)
    plot_dataset_distribution(frame, run_dir)

    train_transform = build_transforms(args.image_size, augment=not args.no_augment)
    eval_transform = build_transforms(args.image_size, augment=False)
    plot_augmentation_examples(frame, train_transform, run_dir)

    train_loader, val_loader, test_loader = make_dataloaders(
        frame,
        train_transform,
        eval_transform,
        args.batch_size,
        args.num_workers,
        args.balance_strategy,
        args.seed,
        device,
        args.sample_level,
    )

    model = build_model(
        args.model,
        args.weights,
        sample_level=args.sample_level,
        num_views=len(view_suffixes),
        group_aggregation=args.group_aggregation,
    ).to(device)
    if args.freeze_backbone_epochs > 0:
        set_backbone_trainable(model, args.model, trainable=False)

    criterion = make_loss(frame[frame["split"] == "train"], args.balance_strategy, device)
    optimizer = make_optimizer(model, args.learning_rate, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=metric_mode(args.model_selection_metric),
        factor=0.35,
        patience=2,
    )

    history: List[Dict[str, float]] = []
    best_score: Optional[float] = None
    epochs_without_improvement = 0
    best_checkpoint = run_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_backbone_epochs + 1 and args.freeze_backbone_epochs > 0:
            print("Unfreezing backbone for full fine tuning.")
            set_backbone_trainable(model, args.model, trainable=True)
            optimizer = make_optimizer(model, args.learning_rate, args.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=metric_mode(args.model_selection_metric),
                factor=0.35,
                patience=2,
            )

        started = time.time()
        train_loss, train_labels, train_probs = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_labels, val_probs = run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )

        train_metrics = compute_metrics(
            train_labels, train_probs, threshold=args.fixed_threshold
        )
        val_metrics = compute_metrics(
            val_labels, val_probs, threshold=args.fixed_threshold
        )
        selected = selection_score(
            args.model_selection_metric, val_loss, val_metrics
        )
        scheduler.step(selected)

        row = {
            "epoch": float(epoch),
            "learning_rate": current_lr(optimizer),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "train_auroc": train_metrics["auroc"],
            "train_auprc": train_metrics["auprc"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "epoch_seconds": time.time() - started,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        improved = is_improvement(best_score, selected, args.model_selection_metric)
        if improved:
            best_score = selected
            epochs_without_improvement = 0
            save_checkpoint(
                best_checkpoint,
                model,
                args,
                threshold=args.fixed_threshold,
                class_to_idx=CLASS_TO_IDX,
            )
        else:
            epochs_without_improvement += 1

        print_epoch_summary(
            epoch,
            args.epochs,
            train_loss,
            val_loss,
            val_metrics,
            selected,
            best_score,
            args.model_selection_metric,
        )

        if epochs_without_improvement >= args.early_stopping_patience:
            print(
                "Early stopping: no validation improvement for "
                f"{args.early_stopping_patience} epochs."
            )
            break

    plot_training_curves(pd.DataFrame(history), run_dir)

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_loss, val_labels, val_probs = run_epoch(
        model, val_loader, criterion, device, optimizer=None
    )
    if args.threshold_policy == "f1":
        threshold = tune_threshold_for_f1(val_labels, val_probs)
    else:
        threshold = args.fixed_threshold

    test_loss, test_labels, test_probs = run_epoch(
        model, test_loader, criterion, device, optimizer=None
    )
    test_metrics = compute_metrics(test_labels, test_probs, threshold=threshold)
    test_preds = (test_probs >= threshold).astype(int)

    save_test_artifacts(
        model,
        best_checkpoint,
        checkpoint,
        threshold,
        test_loss,
        test_labels,
        test_probs,
        test_preds,
        test_loader,
        test_metrics,
        run_dir,
    )

    print("\nFinal test metrics")
    print(json.dumps(test_metrics, indent=2, sort_keys=True))
    print(f"Decision threshold: {threshold:.4f}")
    print(f"Artifacts saved in: {run_dir}")


def evaluate_command(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})

    model_name = checkpoint_args.get("model", args.model)
    image_size = int(checkpoint_args.get("image_size", args.image_size))
    sample_level = checkpoint_args.get("sample_level", "image")
    group_view_suffixes = parse_group_view_suffixes(
        checkpoint_args.get("group_view_suffixes", ",".join(DEFAULT_GROUP_VIEW_SUFFIXES))
    )
    group_aggregation = checkpoint_args.get("group_aggregation", "concat")
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(checkpoint.get("threshold", 0.5))
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else checkpoint_path.parent / f"eval_{args.split}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    split_csv = Path(args.split_csv) if args.split_csv else checkpoint_path.parent / "split.csv"
    if not split_csv.exists():
        raise FileNotFoundError(
            f"Could not find split CSV at {split_csv}. Pass --split-csv explicitly."
        )

    frame = pd.read_csv(split_csv)
    if args.split != "all":
        frame = frame[frame["split"] == args.split].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No rows found for split '{args.split}'.")

    dataset = make_dataset(frame, build_transforms(image_size, augment=False), sample_level)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        model_name,
        "none",
        sample_level=sample_level,
        num_views=len(group_view_suffixes),
        group_aggregation=group_aggregation,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.BCEWithLogitsLoss()
    loss, labels, probs = run_epoch(model, loader, criterion, device, optimizer=None)
    preds = (probs >= threshold).astype(int)
    metrics = compute_metrics(labels, probs, threshold=threshold)
    metrics["loss"] = float(loss)

    write_json(output_dir / "metrics.json", metrics)
    write_predictions_csv(output_dir / "predictions.csv", loader, labels, probs, preds)
    write_classification_report(output_dir / "classification_report.txt", labels, preds)
    plot_evaluation_suite(labels, probs, preds, threshold, output_dir)
    plot_sample_predictions(output_dir / "sample_predictions.png", loader, labels, probs, preds)

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Evaluation artifacts saved in: {output_dir}")


def predict_command(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    model_name = checkpoint_args.get("model", args.model)
    image_size = int(checkpoint_args.get("image_size", args.image_size))
    sample_level = checkpoint_args.get("sample_level", "image")
    group_view_suffixes = parse_group_view_suffixes(
        checkpoint_args.get("group_view_suffixes", ",".join(DEFAULT_GROUP_VIEW_SUFFIXES))
    )
    group_aggregation = checkpoint_args.get("group_aggregation", "concat")
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(checkpoint.get("threshold", 0.5))
    )

    model = build_model(
        model_name,
        "none",
        sample_level=sample_level,
        num_views=len(group_view_suffixes),
        group_aggregation=group_aggregation,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = build_transforms(image_size, augment=False)
    rows = []
    with torch.no_grad():
        if sample_level == "group":
            paths = order_group_paths([Path(path) for path in args.images], group_view_suffixes)
            tensors = []
            for path in paths:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    tensors.append(transform(image))
            tensor = torch.stack(tensors).unsqueeze(0).to(device)
            probability = torch.sigmoid(model(tensor)).item()
            label = "defective" if probability >= threshold else "good"
            rows.append(
                {
                    "path": json.dumps([str(path) for path in paths]),
                    "defect_probability": probability,
                    "prediction": label,
                    "threshold": threshold,
                }
            )
            print(
                f"{', '.join(str(path) for path in paths)}: "
                f"{label} (defect probability {probability:.4f})"
            )
        else:
            for image_path in args.images:
                path = Path(image_path)
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    tensor = transform(image).unsqueeze(0).to(device)
                probability = torch.sigmoid(model(tensor)).item()
                label = "defective" if probability >= threshold else "good"
                rows.append(
                    {
                        "path": str(path),
                        "defect_probability": probability,
                        "prediction": label,
                        "threshold": threshold,
                    }
                )
                print(f"{path}: {label} (defect probability {probability:.4f})")

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Predictions saved to {args.output_csv}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(preference: str) -> torch.device:
    if preference == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.backends.mps.is_built():
            print(
                "MPS support is built into PyTorch, but it is not available in "
                "this session. Falling back to CPU."
            )
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if preference == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but torch.backends.mps is not available.")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(preference)


def make_run_dir(output_dir: Path, run_name: Optional[str], model_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or f"{stamp}_{model_name}"
    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def resolve_class_dirs(
    data_dir: Path,
    good_dir_args: Optional[Sequence[str]],
    defect_dir_args: Optional[Sequence[str]],
) -> Tuple[List[Path], List[Path]]:
    data_dir = data_dir.expanduser()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if good_dir_args:
        good_dirs = [resolve_dir(value, data_dir) for value in good_dir_args]
    else:
        good_dirs = auto_dirs(data_dir, "BUONE")

    if defect_dir_args:
        defect_dirs = [resolve_dir(value, data_dir) for value in defect_dir_args]
    else:
        defect_dirs = auto_dirs(data_dir, "SCARTI")

    if not good_dirs:
        raise ValueError("No good folders found. Pass --good-dir.")
    if not defect_dirs:
        raise ValueError("No defective folders found. Pass --defect-dir.")

    return good_dirs, defect_dirs


def resolve_dir(value: str, data_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        path = data_dir / value
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Image directory not found: {value}")
    return path


def auto_dirs(data_dir: Path, token: str) -> List[Path]:
    token = token.upper()
    return sorted(
        path
        for path in data_dir.iterdir()
        if path.is_dir() and token in path.name.upper()
    )


def parse_group_view_suffixes(value: str) -> List[str]:
    suffixes = [suffix.strip() for suffix in value.split(",") if suffix.strip()]
    if not suffixes:
        raise ValueError("--group-view-suffixes must contain at least one suffix.")
    return suffixes


def order_group_paths(paths: Sequence[Path], view_suffixes: Sequence[str]) -> List[Path]:
    by_suffix: Dict[str, Path] = {}
    for path in paths:
        suffix = view_suffix(path)
        if suffix in by_suffix:
            raise ValueError(f"Duplicate image for grouped suffix '{suffix}': {path}")
        by_suffix[suffix] = path

    missing = [suffix for suffix in view_suffixes if suffix not in by_suffix]
    extra = [str(path) for suffix, path in by_suffix.items() if suffix not in view_suffixes]
    if missing:
        raise ValueError(
            "Grouped prediction is missing required suffixes: " + ", ".join(missing)
        )
    if extra:
        raise ValueError(
            "Grouped prediction received images outside --group-view-suffixes: "
            + ", ".join(extra)
        )
    return [by_suffix[suffix] for suffix in view_suffixes]


def bottle_prefix(path: Path) -> str:
    return path.stem.split("_")[0]


def view_suffix(path: Path) -> str:
    parts = path.name.split("_", 1)
    return parts[1] if len(parts) == 2 else path.name


def scan_dataset(good_dirs: Sequence[Path], defect_dirs: Sequence[Path]) -> pd.DataFrame:
    records = []
    for label_name, directories in [("good", good_dirs), ("defective", defect_dirs)]:
        for directory in directories:
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    group_base = bottle_prefix(path)
                    records.append(
                        {
                            "path": str(path),
                            "label": CLASS_TO_IDX[label_name],
                            "label_name": label_name,
                            "group_id": f"{label_name}:{group_base}",
                            "bottle_prefix": group_base,
                            "view_suffix": view_suffix(path),
                            "source_dir": directory.name,
                        }
                    )

    if not records:
        raise ValueError("No image files were found.")

    frame = pd.DataFrame(records)
    counts = frame["label_name"].value_counts().to_dict()
    if len(counts) != 2:
        raise ValueError(f"Expected exactly two classes, found: {counts}")

    print("Dataset counts:")
    for label_name in ["good", "defective"]:
        print(f"  {label_name}: {counts.get(label_name, 0)}")
    return frame


def make_bottle_group_frame(
    image_frame: pd.DataFrame, view_suffixes: Sequence[str]
) -> pd.DataFrame:
    expected = list(view_suffixes)
    expected_set = set(expected)
    eligible = image_frame[image_frame["view_suffix"].isin(expected_set)].copy()
    ignored = len(image_frame) - len(eligible)
    if eligible.empty:
        raise ValueError(
            "No images matched the grouped view suffixes: " + ", ".join(expected)
        )

    records = []
    incomplete = 0
    duplicate_groups: List[str] = []
    for group_id, group_frame in eligible.groupby("group_id", sort=True):
        by_suffix: Dict[str, List[pd.Series]] = defaultdict(list)
        for _, row in group_frame.iterrows():
            by_suffix[str(row["view_suffix"])].append(row)

        missing = [suffix for suffix in expected if suffix not in by_suffix]
        duplicate_suffixes = [
            suffix for suffix, rows in by_suffix.items() if len(rows) > 1
        ]
        if duplicate_suffixes:
            duplicate_groups.append(f"{group_id}: {duplicate_suffixes}")
            continue
        if missing:
            incomplete += 1
            continue

        ordered_rows = [by_suffix[suffix][0] for suffix in expected]
        label_values = {int(row["label"]) for row in ordered_rows}
        if len(label_values) != 1:
            raise ValueError(f"Grouped bottle has mixed labels: {group_id}")

        paths = [str(row["path"]) for row in ordered_rows]
        source_dirs = sorted({str(row["source_dir"]) for row in ordered_rows})
        records.append(
            {
                "path": paths[0],
                "paths_json": json.dumps(paths),
                "label": int(ordered_rows[0]["label"]),
                "label_name": str(ordered_rows[0]["label_name"]),
                "group_id": group_id,
                "bottle_prefix": str(ordered_rows[0]["bottle_prefix"]),
                "view_suffixes": ",".join(expected),
                "num_views": len(paths),
                "source_dir": "+".join(source_dirs),
            }
        )

    if duplicate_groups:
        examples = "; ".join(duplicate_groups[:5])
        raise ValueError(
            "Some bottle prefixes have duplicate images for the same suffix. "
            f"Examples: {examples}"
        )

    if not records:
        raise ValueError("No complete grouped bottle samples were found.")

    frame = pd.DataFrame(records)
    counts = frame["label_name"].value_counts().to_dict()
    print("Grouped bottle counts:")
    for label_name in ["good", "defective"]:
        print(f"  {label_name}: {counts.get(label_name, 0)}")
    print(f"Grouped views: {', '.join(expected)}")
    print(f"Ignored images with non-group suffixes: {ignored}")
    print(f"Incomplete bottle groups skipped: {incomplete}")
    return frame


def limit_per_class(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    sampled = []
    for _, class_frame in frame.groupby("label", sort=True):
        sampled.append(
            class_frame.sample(n=min(limit, len(class_frame)), random_state=seed)
        )
    return pd.concat(sampled, ignore_index=True)


def make_group_split(
    frame: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Train, validation, and test ratios must sum to 1.0.")

    rng = random.Random(seed)
    assignments: Dict[str, str] = {}

    for label in sorted(frame["label"].unique()):
        class_frame = frame[frame["label"] == label]
        group_ids = list(class_frame["group_id"].drop_duplicates())
        rng.shuffle(group_ids)

        n_groups = len(group_ids)
        if n_groups < 3:
            raise ValueError(
                "Each class needs at least 3 bottle groups to create "
                "train/val/test splits."
            )

        n_train = max(1, int(round(train_ratio * n_groups)))
        n_val = max(1, int(round(val_ratio * n_groups)))
        if n_train + n_val >= n_groups:
            n_val = max(1, n_groups - n_train - 1)
        if n_train + n_val >= n_groups:
            n_train = max(1, n_groups - n_val - 1)
        n_test = n_groups - n_train - n_val
        if n_test < 1:
            raise ValueError(
                "Could not allocate at least one group per split. "
                f"Class {label} has {n_groups} groups."
            )

        for index, group_id in enumerate(group_ids):
            if index < n_train:
                split = "train"
            elif index < n_train + n_val:
                split = "val"
            else:
                split = "test"
            assignments[group_id] = split

    result = frame.copy()
    result["split"] = result["group_id"].map(assignments)
    validate_split(result)
    return result.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def validate_split(frame: pd.DataFrame) -> None:
    missing = []
    for split in ["train", "val", "test"]:
        labels = set(frame.loc[frame["split"] == split, "label_name"].unique())
        if labels != {"good", "defective"}:
            missing.append(f"{split}: {sorted(labels)}")
    if missing:
        raise ValueError(
            "Each split must contain both classes. Got " + "; ".join(missing)
        )


def write_dataset_summary(frame: pd.DataFrame, run_dir: Path) -> None:
    split_counts = (
        frame.groupby(["split", "label_name"]).size().unstack(fill_value=0).sort_index()
    )
    source_counts = (
        frame.groupby(["split", "source_dir", "label_name"])
        .size()
        .rename("count")
        .reset_index()
    )
    split_counts.to_csv(run_dir / "split_summary.csv")
    source_counts.to_csv(run_dir / "source_summary.csv", index=False)
    print("\nSplit counts:")
    print(split_counts)


def build_transforms(image_size: int, augment: bool) -> transforms.Compose:
    interpolation = transforms.InterpolationMode.BICUBIC
    steps: List[object] = [
        PadToSquare(fill="median"),
        transforms.Resize((image_size, image_size), interpolation=interpolation),
    ]

    if augment:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=5,
                    translate=(0.03, 0.03),
                    scale=(0.95, 1.05),
                    shear=(-2, 2),
                    interpolation=interpolation,
                    fill=0,
                ),
                transforms.ColorJitter(brightness=0.12, contrast=0.18),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))],
                    p=0.12,
                ),
            ]
        )

    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


def make_dataloaders(
    frame: pd.DataFrame,
    train_transform: transforms.Compose,
    eval_transform: transforms.Compose,
    batch_size: int,
    num_workers: int,
    balance_strategy: str,
    seed: int,
    device: torch.device,
    sample_level: str,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_frame = frame[frame["split"] == "train"].reset_index(drop=True)
    val_frame = frame[frame["split"] == "val"].reset_index(drop=True)
    test_frame = frame[frame["split"] == "test"].reset_index(drop=True)

    train_dataset = make_dataset(train_frame, train_transform, sample_level)
    val_dataset = make_dataset(val_frame, eval_transform, sample_level)
    test_dataset = make_dataset(test_frame, eval_transform, sample_level)

    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = None
    shuffle = True
    if balance_strategy in {"sampler", "both"}:
        sampler = make_weighted_sampler(train_frame, generator)
        shuffle = False

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=shuffle,
        sampler=sampler,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def make_dataset(
    frame: pd.DataFrame, transform: transforms.Compose, sample_level: str
) -> Dataset:
    if sample_level == "group":
        return GroupedBottleDataset(frame, transform)
    if sample_level == "image":
        return GlassBottleDataset(frame, transform)
    raise ValueError(f"Unknown sample level: {sample_level}")


def make_weighted_sampler(
    train_frame: pd.DataFrame, generator: torch.Generator
) -> WeightedRandomSampler:
    counts = train_frame["label"].value_counts().to_dict()
    weights = train_frame["label"].map(lambda label: 1.0 / counts[int(label)]).values
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def build_model(
    model_name: str,
    weights_mode: str,
    sample_level: str = "image",
    num_views: int = 1,
    group_aggregation: str = "concat",
) -> nn.Module:
    use_weights = weights_mode == "imagenet"
    try:
        return _build_model(
            model_name,
            use_weights,
            sample_level=sample_level,
            num_views=num_views,
            group_aggregation=group_aggregation,
        )
    except Exception as exc:
        if use_weights:
            print(
                "Could not load ImageNet weights. Falling back to random init. "
                f"Reason: {exc}"
            )
            return _build_model(
                model_name,
                False,
                sample_level=sample_level,
                num_views=num_views,
                group_aggregation=group_aggregation,
            )
        raise


def _build_model(
    model_name: str,
    use_weights: bool,
    sample_level: str,
    num_views: int,
    group_aggregation: str,
) -> nn.Module:
    if sample_level == "group":
        return _build_multiview_model(
            model_name,
            use_weights,
            num_views=num_views,
            aggregation=group_aggregation,
        )

    if model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if use_weights else None
        model = efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=0.30), nn.Linear(in_features, 1))
        return model

    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if use_weights else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
        return model

    if model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if use_weights else None
        model = mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 1)
        return model

    raise ValueError(f"Unknown model: {model_name}")


def _build_multiview_model(
    model_name: str, use_weights: bool, num_views: int, aggregation: str
) -> nn.Module:
    if num_views < 1:
        raise ValueError("Grouped bottle model needs at least one view.")

    if model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if use_weights else None
        model = efficientnet_b0(weights=weights)
        feature_dim = model.classifier[1].in_features
        backbone = nn.Sequential(model.features, model.avgpool, nn.Flatten())
    elif model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if use_weights else None
        model = resnet18(weights=weights)
        feature_dim = model.fc.in_features
        backbone = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    elif model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if use_weights else None
        model = mobilenet_v3_small(weights=weights)
        feature_dim = model.classifier[-1].in_features
        backbone = nn.Sequential(model.features, model.avgpool, nn.Flatten())
    else:
        raise ValueError(f"Unknown model: {model_name}")

    classifier_input_dim = feature_dim * num_views if aggregation == "concat" else feature_dim
    classifier = nn.Sequential(
        nn.Dropout(p=0.30),
        nn.Linear(classifier_input_dim, 1),
    )
    return MultiViewClassifier(
        backbone=backbone,
        feature_dim=feature_dim,
        num_views=num_views,
        aggregation=aggregation,
        classifier=classifier,
    )


def set_backbone_trainable(model: nn.Module, model_name: str, trainable: bool) -> None:
    if isinstance(model, MultiViewClassifier):
        for parameter in model.parameters():
            parameter.requires_grad = trainable
        if not trainable:
            for parameter in model.classifier.parameters():
                parameter.requires_grad = True
        return

    for parameter in model.parameters():
        parameter.requires_grad = trainable

    if not trainable:
        if model_name in {"efficientnet_b0", "mobilenet_v3_small"}:
            for parameter in model.classifier.parameters():
                parameter.requires_grad = True
        elif model_name == "resnet18":
            for parameter in model.fc.parameters():
                parameter.requires_grad = True


def make_loss(
    train_frame: pd.DataFrame, balance_strategy: str, device: torch.device
) -> nn.Module:
    if balance_strategy in {"loss", "both"}:
        counts = train_frame["label"].value_counts().to_dict()
        negatives = counts.get(CLASS_TO_IDX["good"], 0)
        positives = counts.get(CLASS_TO_IDX["defective"], 0)
        if positives == 0:
            raise ValueError("No defective samples in train split.")
        pos_weight = torch.tensor([negatives / positives], dtype=torch.float32).to(device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return nn.BCEWithLogitsLoss()


def make_optimizer(
    model: nn.Module, learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters found.")
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
) -> Tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)

    all_labels: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    total_loss = 0.0
    total_items = 0

    progress = tqdm(loader, leave=False, desc="train" if training else "eval")
    for images, labels, _ in progress:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda").unsqueeze(1)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
        probs = torch.sigmoid(logits.detach()).cpu().numpy().reshape(-1)
        all_probs.append(probs)
        all_labels.append(labels.detach().cpu().numpy().reshape(-1))
        progress.set_postfix(loss=total_loss / max(total_items, 1))

    labels_np = np.concatenate(all_labels).astype(int)
    probs_np = np.concatenate(all_probs).astype(float)
    return total_loss / max(total_items, 1), labels_np, probs_np


def compute_metrics(
    labels: np.ndarray, probs: np.ndarray, threshold: float
) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "precision": float(
            precision_score(labels, preds, zero_division=0)
        ),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "threshold": float(threshold),
    }
    if len(np.unique(labels)) == 2:
        metrics["auroc"] = float(roc_auc_score(labels, probs))
        metrics["auprc"] = float(average_precision_score(labels, probs))
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    return metrics


def tune_threshold_for_f1(labels: np.ndarray, probs: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    if len(thresholds) == 0:
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index])


def selection_score(
    metric_name: str, val_loss: float, val_metrics: Dict[str, float]
) -> float:
    if metric_name == "loss":
        return float(val_loss)
    return float(val_metrics[metric_name])


def metric_mode(metric_name: str) -> str:
    return "min" if metric_name == "loss" else "max"


def is_improvement(
    best_score: Optional[float], current: float, metric_name: str
) -> bool:
    if best_score is None or math.isnan(best_score):
        return True
    if math.isnan(current):
        return False
    if metric_name == "loss":
        return current < best_score
    return current > best_score


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def print_epoch_summary(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    val_metrics: Dict[str, float],
    selected: float,
    best_score: Optional[float],
    metric_name: str,
) -> None:
    best_text = "nan" if best_score is None else f"{best_score:.4f}"
    print(
        f"Epoch {epoch:03d}/{total_epochs:03d} | "
        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
        f"val_f1={val_metrics['f1']:.4f} val_recall={val_metrics['recall']:.4f} "
        f"val_auprc={val_metrics['auprc']:.4f} val_auroc={val_metrics['auroc']:.4f} | "
        f"{metric_name}={selected:.4f} best={best_text}"
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    threshold: float,
    class_to_idx: Dict[str, int],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "threshold": float(threshold),
            "class_to_idx": class_to_idx,
        },
        path,
    )


def save_test_artifacts(
    model: nn.Module,
    best_checkpoint: Path,
    checkpoint: Dict[str, object],
    threshold: float,
    test_loss: float,
    test_labels: np.ndarray,
    test_probs: np.ndarray,
    test_preds: np.ndarray,
    test_loader: DataLoader,
    test_metrics: Dict[str, float],
    run_dir: Path,
) -> None:
    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["threshold"] = float(threshold)
    checkpoint["test_metrics"] = test_metrics
    torch.save(checkpoint, best_checkpoint)

    metrics_with_loss = dict(test_metrics)
    metrics_with_loss["loss"] = float(test_loss)
    write_json(run_dir / "test_metrics.json", metrics_with_loss)
    write_predictions_csv(
        run_dir / "test_predictions.csv",
        test_loader,
        test_labels,
        test_probs,
        test_preds,
    )
    write_classification_report(
        run_dir / "classification_report.txt", test_labels, test_preds
    )
    plot_evaluation_suite(test_labels, test_probs, test_preds, threshold, run_dir)
    plot_sample_predictions(
        run_dir / "sample_predictions.png",
        test_loader,
        test_labels,
        test_probs,
        test_preds,
    )


def write_predictions_csv(
    path: Path,
    loader: DataLoader,
    labels: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
) -> None:
    paths: List[str] = []
    for _, _, batch_paths in loader:
        paths.extend(list(batch_paths))

    frame = pd.DataFrame(
        {
            "path": paths,
            "true_label": [IDX_TO_CLASS[int(label)] for label in labels],
            "true_label_idx": labels.astype(int),
            "defect_probability": probs,
            "predicted_label": [IDX_TO_CLASS[int(pred)] for pred in preds],
            "predicted_label_idx": preds.astype(int),
        }
    )
    frame.to_csv(path, index=False)


def write_classification_report(path: Path, labels: np.ndarray, preds: np.ndarray) -> None:
    report = classification_report(
        labels,
        preds,
        target_names=["good", "defective"],
        zero_division=0,
    )
    path.write_text(report)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def plot_dataset_distribution(frame: pd.DataFrame, run_dir: Path) -> None:
    counts = (
        frame.groupby(["split", "label_name"])
        .size()
        .unstack(fill_value=0)
        .reindex(["train", "val", "test"])
    )
    ax = counts.plot(kind="bar", figsize=(8, 5), color=["#4c78a8", "#e45756"])
    ax.set_title("Class distribution by split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Samples")
    ax.legend(title="Class")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(run_dir / "dataset_distribution.png", dpi=160)
    plt.close()

    source_counts = (
        frame.groupby(["source_dir", "label_name"]).size().unstack(fill_value=0)
    )
    ax = source_counts.plot(kind="bar", figsize=(9, 5), color=["#4c78a8", "#e45756"])
    ax.set_title("Samples by source folder")
    ax.set_xlabel("Source folder")
    ax.set_ylabel("Samples")
    ax.legend(title="Class")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(run_dir / "source_distribution.png", dpi=160)
    plt.close()


def plot_augmentation_examples(
    frame: pd.DataFrame, train_transform: transforms.Compose, run_dir: Path
) -> None:
    defective = frame[(frame["split"] == "train") & (frame["label_name"] == "defective")]
    candidates = defective if not defective.empty else frame[frame["split"] == "train"]
    if candidates.empty:
        return

    path = first_sample_path(candidates.iloc[0])
    with Image.open(path) as image:
        image = image.convert("RGB")
        fig, axes = plt.subplots(2, 4, figsize=(10, 5))
        for axis in axes.flat:
            tensor = train_transform(image)
            axis.imshow(tensor_to_image(tensor))
            axis.axis("off")
        fig.suptitle("Training augmentation examples")
        plt.tight_layout()
        plt.savefig(run_dir / "augmentation_examples.png", dpi=160)
        plt.close()


def first_sample_path(row: pd.Series) -> Path:
    if "paths_json" in row and isinstance(row["paths_json"], str):
        paths = json.loads(row["paths_json"])
        if paths:
            return Path(paths[0])
    return Path(row["path"])


def plot_training_curves(history: pd.DataFrame, run_dir: Path) -> None:
    if history.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(history["epoch"], history["val_f1"], label="F1")
    axes[1].plot(history["epoch"], history["val_balanced_accuracy"], label="Balanced acc")
    axes[1].set_title("Validation threshold metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(history["epoch"], history["val_auprc"], label="PR AUC")
    axes[2].plot(history["epoch"], history["val_auroc"], label="ROC AUC")
    axes[2].set_title("Validation ranking metrics")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(run_dir / "training_curves.png", dpi=160)
    plt.close()


def plot_evaluation_suite(
    labels: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    threshold: float,
    run_dir: Path,
) -> None:
    plot_confusion(labels, preds, run_dir / "confusion_matrix.png")
    plot_roc(labels, probs, run_dir / "roc_curve.png")
    plot_precision_recall(labels, probs, threshold, run_dir / "precision_recall_curve.png")
    plot_probability_histogram(
        labels, probs, threshold, run_dir / "probability_histogram.png"
    )


def plot_confusion(labels: np.ndarray, preds: np.ndarray, path: Path) -> None:
    matrix = confusion_matrix(labels, preds, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], labels=["good", "defective"])
    axis.set_yticks([0, 1], labels=["good", "defective"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_roc(labels: np.ndarray, probs: np.ndarray, path: Path) -> None:
    if len(np.unique(labels)) < 2:
        return
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_value = roc_auc_score(labels, probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_precision_recall(
    labels: np.ndarray, probs: np.ndarray, threshold: float, path: Path
) -> None:
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_probability_histogram(
    labels: np.ndarray, probs: np.ndarray, threshold: float, path: Path
) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist(probs[labels == 0], bins=30, alpha=0.7, label="good", color="#4c78a8")
    plt.hist(
        probs[labels == 1],
        bins=30,
        alpha=0.7,
        label="defective",
        color="#e45756",
    )
    plt.axvline(threshold, color="black", linestyle="--", label=f"threshold {threshold:.2f}")
    plt.xlabel("Predicted defect probability")
    plt.ylabel("Images")
    plt.title("Prediction probability distribution")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_sample_predictions(
    path: Path,
    loader: DataLoader,
    labels: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
) -> None:
    paths: List[str] = []
    for _, _, batch_paths in loader:
        paths.extend(list(batch_paths))

    indices = select_interesting_indices(labels, probs, preds, max_items=16)
    if not indices:
        return

    cols = 4
    rows = math.ceil(len(indices) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.1, rows * 3.2))
    axes_array = np.array(axes).reshape(-1)

    for axis, index in zip(axes_array, indices):
        image = make_prediction_preview(parse_identifier_paths(paths[index]))
        axis.imshow(image, cmap="gray")
        true_name = IDX_TO_CLASS[int(labels[index])]
        pred_name = IDX_TO_CLASS[int(preds[index])]
        axis.set_title(
            f"true={true_name}\npred={pred_name} p={probs[index]:.2f}",
            fontsize=9,
        )
        axis.axis("off")

    for axis in axes_array[len(indices) :]:
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def parse_identifier_paths(identifier: str) -> List[Path]:
    try:
        parsed = json.loads(identifier)
        if isinstance(parsed, list):
            return [Path(path) for path in parsed]
    except json.JSONDecodeError:
        pass
    return [Path(identifier)]


def make_prediction_preview(paths: Sequence[Path]) -> Image.Image:
    images = []
    for path in paths:
        with Image.open(path) as image:
            preview = image.convert("L")
            preview.thumbnail((180, 180))
            images.append(preview.copy())

    if len(images) == 1:
        return images[0]

    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("L", (width, height), 0)
    offset = 0
    for image in images:
        top = (height - image.height) // 2
        canvas.paste(image, (offset, top))
        offset += image.width
    return canvas


def select_interesting_indices(
    labels: np.ndarray, probs: np.ndarray, preds: np.ndarray, max_items: int
) -> List[int]:
    false_negative = np.where((labels == 1) & (preds == 0))[0]
    false_positive = np.where((labels == 0) & (preds == 1))[0]
    true_positive = np.where((labels == 1) & (preds == 1))[0]
    true_negative = np.where((labels == 0) & (preds == 0))[0]

    ordered: List[int] = []
    ordered.extend(false_negative[np.argsort(probs[false_negative])].tolist())
    ordered.extend(false_positive[np.argsort(-probs[false_positive])].tolist())
    ordered.extend(true_positive[np.argsort(-probs[true_positive])].tolist())
    ordered.extend(true_negative[np.argsort(probs[true_negative])].tolist())

    unique: List[int] = []
    seen = set()
    for index in ordered:
        if index not in seen:
            unique.append(int(index))
            seen.add(index)
        if len(unique) >= max_items:
            break
    return unique


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().cpu().clone()
    for channel, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        image[channel] = image[channel] * std + mean
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return image


if __name__ == "__main__":
    main()
