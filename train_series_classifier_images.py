#!/usr/bin/env python3
"""
train_series_classifier_images.py - Train with JoyTag backbone on images using frozen epochs,
then minimally unfreeze the last block(s). Produces the same analysis artifacts as the
embedding-based trainer, saved to a separate output directory for comparison.

Usage:
  python train_series_classifier_images.py [dataset_file]
    --frozen-epochs 3 --unfrozen-epochs 10 --unfreeze-last-k 1 \
    --batch-size 32 --lr-head 1e-3 --lr-backbone 1e-4 \
    --balance-method weighted --output-dir series_classifier_output_images
"""

import os
import sys
import json
import glob
import math
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import contextlib
import torch.backends.cudnn as cudnn
import time
import platform

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class SeriesClassifier(nn.Module):
    """Multi-label classifier head for series tags."""

    def __init__(self, embedding_dim: int = 768, num_classes: int = 0, hidden_dims: List[int] = [512, 256], dropout: float = 0.3):
        super(SeriesClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        layers: List[nn.Module] = []
        prev_dim = embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)


def find_most_recent_processed_dataset():
    patterns = ['dataset_joytag_complete_*.json', 'advanced_joytag_complete_*.json']
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    candidates = sorted(files, key=os.path.getmtime, reverse=True)
    return candidates[0] if candidates else None


def load_processed_dataset(dataset_file):
    print(f"📚 Loading processed dataset: {dataset_file}")
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = data.get('results', [])
    print(f"✅ Loaded {len(results)} results from dataset")
    samples = []
    unlabeled_count = 0
    for r in results:
        fp = r.get('file_path')
        if fp and os.path.exists(fp):
            tags = r.get('existing_series_tags') or []
            h = r.get('hash')
            # Train ONLY on labeled samples (at least one series tag)
            if len(tags) >= 1:
                samples.append((fp, tags, h))
            else:
                unlabeled_count += 1
    print(f"📊 Usable labeled samples: {len(samples)} (skipped {unlabeled_count} unlabeled)")
    return samples


def create_label_encoder(series_tags_list: List[List[str]]):
    all_tags = set()
    for tags in series_tags_list:
        all_tags.update(tags)
    sorted_tags = sorted(list(all_tags))
    label_encoder = {tag: idx for idx, tag in enumerate(sorted_tags)}
    print(f"🏷️  Found {len(sorted_tags)} unique series tags")
    print(f"   Sample tags: {sorted_tags[:10]}")
    return label_encoder, sorted_tags


def encode_labels(series_tags_list: List[List[str]], label_encoder: dict) -> np.ndarray:
    num_classes = len(label_encoder)
    labels = []
    for tags in series_tags_list:
        vec = np.zeros(num_classes, dtype=np.float32)
        for t in tags:
            if t in label_encoder:
                vec[label_encoder[t]] = 1.0
        labels.append(vec)
    return np.array(labels)


def calculate_class_weights(labels: np.ndarray):
    num_classes = labels.shape[1]
    class_counts = np.sum(labels, axis=0)
    total = len(labels)
    # Avoid divide-by-zero: if a class has zero positives, set weight to 1.0 (neutral)
    safe_counts = np.where(class_counts > 0, class_counts, 1.0)
    weights = total / (num_classes * safe_counts)
    weights = np.clip(weights, 0.1, 10.0)
    # For zero-count classes, force weight to 1.0
    weights = np.where(class_counts > 0, weights, 1.0)
    zero_classes = int(np.sum(class_counts == 0))
    if zero_classes > 0:
        print(f"⚠️  {zero_classes} classes have zero positives; using neutral weight 1.0 for those.")
    return weights


class ImageSeriesDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, List[str], Optional[str]]], labels: np.ndarray, augment: bool, cache_dir: Optional[Path] = None, input_size: int = 448):
        # samples: list of (file_path, tags, hash)
        self.samples = samples
        self.labels = torch.FloatTensor(labels)
        # JoyTag preprocess: pad to square -> resize 448 -> to_tensor -> normalize
        self.base_tf = T.Compose([])  # we will apply steps manually to match exact pipeline
        self.augment = augment
        self.input_size = int(input_size)
        self.color_jitter = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02)
        # geometric augmentations (no horizontal flips; left/right is important)
        self.rotate = T.RandomRotation(degrees=5)
        self.affine = T.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05), shear=5)
        self.persp = T.RandomPerspective(distortion_scale=0.08, p=0.2)
        self.cache_dir = cache_dir
        self.use_cache = cache_dir is not None
        if self.use_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.samples)

    def _preprocess_image(self, img: Image.Image) -> torch.Tensor:
        w, h = img.size
        max_dim = max(w, h)
        padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))
        if max_dim != self.input_size:
            padded = padded.resize((self.input_size, self.input_size), Image.Resampling.BICUBIC)
        t = T.functional.to_tensor(padded)
        t = T.functional.normalize(
            t,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )
        return t

    def __getitem__(self, idx):
        path, _tags, h = self.samples[idx]
        y = self.labels[idx]
        # Cache key
        cache_key = h if h is not None else str(hash(path))
        # Use cache only when not augmenting (aug should vary per epoch)
        if self.use_cache and not self.augment:
            pt_path = self.cache_dir / f"{cache_key}.pt"
            if pt_path.exists():
                try:
                    tensor = torch.load(pt_path)
                    if isinstance(tensor, torch.Tensor) and tensor.shape[1:] == (self.input_size, self.input_size):
                        return tensor, y
                except Exception:
                    pass
        img = Image.open(path).convert('RGB')
        if self.augment:
            img = self.color_jitter(img)
            img = self.rotate(img)
            img = self.affine(img)
            img = self.persp(img)
        x = self._preprocess_image(img)
        if self.use_cache and not self.augment:
            try:
                torch.save(x, pt_path)
            except Exception:
                pass
        return x, y


def load_joytag_model(device: torch.device):
    from joytag.Models import ViT
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'joytag')
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(current_dir), 'joytag')
    model = ViT.load_model(model_path)
    model = model.to(device)
    model.train()
    return model


def freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def set_minimal_unfreeze_vit(model, last_k: int):
    freeze_all(model)
    # Unfreeze last K transformer blocks + norm
    if hasattr(model, 'blocks'):
        for p in model.blocks[-last_k:].parameters():
            p.requires_grad = True
    if hasattr(model, 'norm'):
        for p in model.norm.parameters():
            p.requires_grad = True


def forward_backbone_features(model, images: torch.Tensor, trainable_last_k: int, device: torch.device):
    B, C, H, W = images.shape
    x = model.patch_embeddings(images)
    x = x.flatten(2).transpose(1, 2)
    x = model.pos_embedding(x, W, H)
    total_blocks = len(model.blocks)
    frozen_blocks = total_blocks - trainable_last_k
    with torch.no_grad():
        for i in range(frozen_blocks):
            x = model.blocks[i](x)
    for i in range(frozen_blocks, total_blocks):
        x = model.blocks[i](x)
    x = model.norm(x)
    x = x.mean(dim=1)
    return x


def extract_features(backbone, inputs: torch.Tensor, trainable_last_k: int, device: torch.device) -> torch.Tensor:
    if inputs.dim() == 2:
        return inputs
    return forward_backbone_features(backbone, inputs, trainable_last_k, device)


def train_one_epoch(backbone, head, loader, optimizer, criterion, device, trainable_last_k: int, log_interval: int = 50, limit_batches: int = 0):
    if backbone is not None:
        backbone.train()
    head.train()
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    total = 0.0
    count = 0
    total_batches = len(loader)
    for batch_idx, (inputs, labels) in enumerate(loader):
        # Skip single-item batches to avoid BatchNorm errors in training
        if labels.shape[0] < 2:
            if log_interval and (batch_idx + 1) % max(1, log_interval) == 0:
                print("   [train] skipping batch with size 1 to avoid BatchNorm error")
            continue
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        ctx = torch.amp.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else contextlib.nullcontext()
        with ctx:
            feats = extract_features(backbone, inputs, trainable_last_k, device)
            logits = head(feats)
            loss = criterion(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += float(loss.item())
        count += 1
        if log_interval and (batch_idx + 1) % log_interval == 0:
            print(f"   [train] batch {batch_idx+1}/{total_batches} avg_loss={total/max(1,count):.4f}")
        if limit_batches and (batch_idx + 1) >= limit_batches:
            break
    return total / max(1, count)


@torch.no_grad()
def validate_one_epoch(backbone, head, loader, criterion, device, trainable_last_k: int, log_interval: int = 0, limit_batches: int = 0):
    if backbone is not None:
        backbone.eval()
    head.eval()
    total = 0.0
    count = 0
    preds_all = []
    labels_all = []
    probs_all = []
    total_batches = len(loader)
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        ctx = torch.amp.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else contextlib.nullcontext()
        with ctx:
            feats = extract_features(backbone, inputs, trainable_last_k, device)
            logits = head(feats)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float().cpu().numpy()
        total += float(loss.item())
        count += 1
        preds_all.append(preds)
        labels_all.append(labels.cpu().numpy())
        probs_all.append(probs.detach().cpu().numpy())
        if log_interval and (batch_idx + 1) % log_interval == 0:
            print(f"   [val] batch {batch_idx+1}/{total_batches} avg_loss={total/max(1,count):.4f}")
        if limit_batches and (batch_idx + 1) >= limit_batches:
            break
    preds_all = np.concatenate(preds_all, axis=0) if preds_all else np.zeros((0,))
    labels_all = np.concatenate(labels_all, axis=0) if labels_all else np.zeros((0,))
    probs_all = np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0,))
    return total / max(1, count), preds_all, labels_all, probs_all


def main():
    parser = argparse.ArgumentParser(description='Train series classifier on images with minimal unfreezing')
    parser.add_argument('dataset_file', nargs='?', default=None, help='Path to processed dataset JSON (defaults to latest dataset_joytag_complete_*.json)')
    parser.add_argument('--frozen-epochs', type=int, default=3, help='Epochs with fully frozen backbone (train head only)')
    parser.add_argument('--unfrozen-epochs', type=int, default=10, help='Epochs with minimal unfreezing')
    parser.add_argument('--unfreeze-last-k', type=int, default=1, help='Number of last ViT blocks to unfreeze')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr-head', type=float, default=1e-3, help='Learning rate for series head')
    parser.add_argument('--lr-backbone', type=float, default=1e-4, help='Learning rate for unfrozen backbone blocks')
    parser.add_argument('--balance-method', choices=['auto', 'none', 'weighted', 'focal'], default='auto', help='Loss balancing method (auto selects based on class imbalance)')
    parser.add_argument('--output-dir', default='series_classifier_output_images', help='Output directory')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='Validation ratio (no test set will be created)')
    parser.add_argument('--cache-dir', type=str, default='image_cache_448', help='Optional directory to cache preprocessed 448x448 tensors (.pt)')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader worker processes')
    parser.add_argument('--pin-memory', action='store_true', help='Enable pinned memory for DataLoader')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"🚀 Training (images) with frozen then minimal unfreezing")
    print(f"📁 Output: {out_dir}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")

    dataset_file = args.dataset_file or find_most_recent_processed_dataset()
    if not dataset_file:
        print("❌ No processed dataset files found")
        return
    print(f"📁 Using dataset: {dataset_file}")

    # Load samples and build labels
    samples = load_processed_dataset(dataset_file)
    file_paths = [s[0] for s in samples]
    series_tags_list = [s[1] for s in samples]
    label_encoder, label_names = create_label_encoder(series_tags_list)
    labels = encode_labels(series_tags_list, label_encoder)

    # Splits (train/val only)
    X_train_idx, X_val_idx = train_test_split(list(range(len(samples))), test_size=max(0.01, min(0.9, args.val_ratio)), random_state=42, stratify=None)
    X_train = [samples[i] for i in X_train_idx]
    X_val = [samples[i] for i in X_val_idx]
    y_train = labels[X_train_idx]
    y_val = labels[X_val_idx]
    print(f"📊 Splits: train {len(X_train)} | val {len(X_val)}")

    # Datasets and loaders
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    train_ds_images = ImageSeriesDataset(X_train, y_train, augment=True, cache_dir=cache_dir, input_size=448)
    val_ds_images = ImageSeriesDataset(X_val, y_val, augment=False, cache_dir=cache_dir, input_size=448)

    loader_kwargs = dict(num_workers=max(0, args.num_workers), pin_memory=args.pin_memory)
    train_loader_images = DataLoader(train_ds_images, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader_images = DataLoader(val_ds_images, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    # Models
    head_in_dim = 768
    backbone = load_joytag_model(device) if args.unfrozen_epochs > 0 else None
    head = SeriesClassifier(embedding_dim=head_in_dim, num_classes=len(label_encoder), hidden_dims=[512, 256], dropout=0.3).to(device)

    # Loss
    chosen_balance = args.balance_method
    if chosen_balance == 'auto':
        try:
            counts = np.sum(y_train, axis=0)
            median = float(np.median(counts[counts > 0])) if np.any(counts > 0) else 0.0
            maxc = float(np.max(counts)) if counts.size > 0 else 0.0
            if median > 0 and maxc / median >= 20.0:
                chosen_balance = 'focal'
            else:
                chosen_balance = 'weighted'
        except Exception:
            chosen_balance = 'weighted'

    if chosen_balance == 'weighted':
        class_weights = calculate_class_weights(y_train)
        pos_weights = torch.FloatTensor(class_weights).to(device)
        def crit(logits, targets):
            return nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weights)
        criterion = crit
        print("🔧 Using Weighted BCE Loss")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("🔧 Using standard BCE Loss")

    # Training loop
    train_losses = []
    val_losses = []
    best_val = float('inf')
    best_state = None

    # Frozen stage
    if args.frozen_epochs > 0:
        print(f"\n🔒 Frozen stage: {args.frozen_epochs} epochs")
        if backbone is not None:
            freeze_all(backbone)
        optimizer = optim.Adam(head.parameters(), lr=args.lr_head)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        
        for epoch in range(args.frozen_epochs):
            tr = train_one_epoch(backbone, head, train_loader_images, optimizer, criterion, device, trainable_last_k=0)
            vl, preds_v, labels_v, probs_v = validate_one_epoch(backbone, head, val_loader_images, criterion, device, trainable_last_k=0)
            train_losses.append(tr)
            val_losses.append(vl)
            scheduler.step(vl)
            
            if vl < best_val:
                best_val = vl
                best_state = {
                    'backbone': backbone.state_dict() if backbone is not None else None,
                    'head': head.state_dict(),
                    'epoch': epoch,
                    'stage': 'frozen'
                }
            
            print(f"[Frozen] Epoch {epoch+1}/{args.frozen_epochs} | ValLoss {vl:.4f} | TrainLoss {tr:.4f}")

    # Unfrozen stage
    if args.unfrozen_epochs > 0 and backbone is not None:
        print(f"\n🔓 Unfrozen stage: {args.unfrozen_epochs} epochs")
        set_minimal_unfreeze_vit(backbone, args.unfreeze_last_k)
        params = [
            {'params': [p for p in backbone.parameters() if p.requires_grad], 'lr': args.lr_backbone},
            {'params': head.parameters(), 'lr': args.lr_head},
        ]
        optimizer = optim.Adam(params)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        
        for epoch in range(args.unfrozen_epochs):
            tr = train_one_epoch(backbone, head, train_loader_images, optimizer, criterion, device, trainable_last_k=args.unfreeze_last_k)
            vl, preds_v, labels_v, probs_v = validate_one_epoch(backbone, head, val_loader_images, criterion, device, trainable_last_k=args.unfreeze_last_k)
            train_losses.append(tr)
            val_losses.append(vl)
            scheduler.step(vl)
            
            if vl < best_val:
                best_val = vl
                best_state = {
                    'backbone': backbone.state_dict() if backbone is not None else None,
                    'head': head.state_dict(),
                    'epoch': epoch,
                    'stage': 'unfrozen'
                }
            
            print(f"[Unfrozen] Epoch {epoch+1}/{args.unfrozen_epochs} | ValLoss {vl:.4f} | TrainLoss {tr:.4f}")

    # Restore best
    if best_state is not None:
        if backbone is not None and best_state.get('backbone') is not None:
            backbone.load_state_dict(best_state['backbone'])
        head.load_state_dict(best_state['head'])

    # Final evaluation
    val_eval_loss, val_eval_preds, val_eval_labels, val_eval_probs = validate_one_epoch(backbone, head, val_loader_images, criterion, device, trainable_last_k=args.unfreeze_last_k)
    report = classification_report(val_eval_labels, val_eval_preds, target_names=label_names, zero_division=0, output_dict=True)

    # Save artifacts
    torch.save({
        'epoch': best_state['epoch'] if best_state else -1,
        'backbone_state_dict': (backbone.state_dict() if backbone is not None else None),
        'head_state_dict': head.state_dict(),
        'label_names': label_names,
        'model_config': {
            'embedding_dim': head_in_dim,
            'num_classes': len(label_encoder),
            'hidden_dims': [512, 256],
            'dropout': 0.3,
            'unfreeze_last_k': args.unfreeze_last_k
        }
    }, out_dir / 'best_model.pth')

    # Results JSON
    results = {
        'val_eval_loss': float(val_eval_loss),
        'overall_metrics': {
            'micro_f1': report['micro avg']['f1-score'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
        },
        'per_class_metrics': report,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_epoch': best_state['epoch'] if best_state else -1,
            'best_stage': best_state['stage'] if best_state else 'n/a'
        },
        'model_config': {
            'embedding_dim': head_in_dim,
            'num_classes': len(label_encoder),
            'hidden_dims': [512, 256],
            'dropout': 0.3,
            'unfreeze_last_k': args.unfreeze_last_k
        },
        'training_config': {
            'frozen_epochs': args.frozen_epochs,
            'unfrozen_epochs': args.unfrozen_epochs,
            'unfreeze_last_k': args.unfreeze_last_k,
            'batch_size': args.batch_size,
            'lr_head': args.lr_head,
            'lr_backbone': args.lr_backbone,
            'balance_method': args.balance_method
        }
    }
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n🎉 Training (images) complete!")
    print(f"📁 Results saved to: {out_dir}")


if __name__ == '__main__':
    main()