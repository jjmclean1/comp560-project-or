#!/usr/bin/env python3
"""
COMP560 Object Re-Identification Training Example

Demonstrates how to train a ResNet50-based ReID model using
ArcFace or triplet loss with the provided Parquet datasets, and how to
generate prediction CSV files for submission.

Usage (training):
    python train_example.py --data_root ./datasets/dataset_a --epochs 10
    python train_example.py --data_root ./datasets/dataset_b --loss triplet --epochs 20

Usage (prediction generation):
    python train_example.py --predict --checkpoint ./checkpoints/best_model.pth --dataset_root ./datasets/dataset_a --dataset_name dataset_a --output predictions/dataset_a.csv
    python train_example.py --predict --checkpoint ./checkpoints/best_model.pth --dataset_root ./datasets/dataset_b --dataset_name dataset_b --output predictions/dataset_b.csv

This script loads training data from Parquet files. Students should
design their own training strategies for best performance.
"""

import argparse
import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm


# ============================================================================
# Dataset
# ============================================================================

class ReIDTrainDataset(Dataset):
    """Training dataset for ReID loaded from Parquet metadata."""

    def __init__(self, root: str, parquet_file: str = "train.parquet", image_size=(224, 224)):
        self.root = root
        df = pd.read_parquet(os.path.join(root, parquet_file))

        # Filter to train split only (for datasets with mixed parquet)
        if "split" in df.columns:
            df = df[df["split"] == "train"]

        unique_ids = sorted(df["identity"].unique())
        self.id_to_label = {pid: i for i, pid in enumerate(unique_ids)}
        self.num_classes = len(unique_ids)

        self.image_paths = df["image_path"].tolist()
        self.labels = [self.id_to_label[pid] for pid in df["identity"].values]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.image_paths[idx])
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, self.labels[idx]


# ============================================================================
# Loss Functions
# ============================================================================

class ArcFaceLoss(nn.Module):
    """Additive Angular Margin Loss (ArcFace)."""

    def __init__(self, embedding_dim, num_classes, s=30.0, m=0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))

        one_hot = torch.zeros_like(cosine).scatter_(1, labels.unsqueeze(1), 1.0)
        target_logits = torch.cos(theta + self.m * one_hot)

        logits = target_logits * self.s
        return F.cross_entropy(logits, labels)


class TripletLoss(nn.Module):
    """Online hard triplet mining loss."""

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        dist_mat = 1 - torch.mm(embeddings, embeddings.t())

        labels = labels.unsqueeze(0)
        same_identity = labels == labels.t()

        loss = torch.tensor(0.0, device=embeddings.device)
        count = 0

        for i in range(embeddings.size(0)):
            pos_mask = same_identity[i].clone()
            pos_mask[i] = False
            neg_mask = ~same_identity[i]

            if pos_mask.any() and neg_mask.any():
                hardest_pos = dist_mat[i][pos_mask].max()
                hardest_neg = dist_mat[i][neg_mask].min()
                loss += F.relu(hardest_pos - hardest_neg + self.margin)
                count += 1

        return loss / max(count, 1)


# ============================================================================
# Model
# ============================================================================

class TrainableModel(nn.Module):
    """ResNet50 backbone with configurable embedding head for ReID training."""

    def __init__(self, embedding_dim=512):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(2048, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self):
        return self._embedding_dim

    def forward(self, images):
        features = self.backbone(images)
        return self.head(features)

    def encode(self, images):
        features = self.forward(images)
        return F.normalize(features, p=2, dim=1)


# ============================================================================
# Training Loop
# ============================================================================

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = ReIDTrainDataset(
        args.data_root,
        image_size=(args.image_size, args.image_size),
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    print(f"Training set: {len(dataset)} images, {dataset.num_classes} identities")

    # Model
    model = TrainableModel(embedding_dim=args.embedding_dim).to(device)

    # Loss
    if args.loss == "arcface":
        criterion = ArcFaceLoss(args.embedding_dim, dataset.num_classes).to(device)
    elif args.loss == "triplet":
        criterion = TripletLoss(margin=args.margin)
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    # Optimizer: different LR for backbone (pretrained) vs head (new)
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())
    loss_params = list(criterion.parameters()) if hasattr(criterion, 'parameters') else []

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
        {"params": loss_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)

    # LR Scheduler: cosine annealing with warmup
    total_steps = len(dataloader) * args.epochs
    warmup_steps = len(dataloader) * args.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        if hasattr(criterion, 'train'):
            criterion.train()

        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            embeddings = model(images)
            loss = criterion(embeddings, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'embedding_dim': args.embedding_dim,
            }, save_dir / "best_model.pth")
            print(f"  Saved best model (loss={avg_loss:.4f})")

        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
                'embedding_dim': args.embedding_dim,
            }, save_dir / f"checkpoint_epoch{epoch+1}.pth")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


# ============================================================================
# Prediction Generation
# ============================================================================

class ImageDataset(Dataset):
    """Simple image dataset for inference."""

    def __init__(self, root, image_paths, image_size=(224, 224)):
        self.root = root
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.image_paths[idx])
        image = Image.open(path).convert("RGB")
        return self.transform(image), idx


def encode_images(model, dataset, batch_size, num_workers, device):
    """Encode all images and return L2-normalized embeddings in order."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    emb_list, idx_list = [], []
    with torch.inference_mode():
        for images, indices in tqdm(loader, desc="Encoding"):
            emb = model.encode(images.to(device))
            emb_list.append(emb.cpu().numpy())
            idx_list.append(indices.numpy())
    embeddings = np.vstack(emb_list)
    indices = np.concatenate(idx_list)
    return embeddings[np.argsort(indices)]


def load_query_gallery(root, dataset_name):
    """Load query/gallery paths based on dataset name."""
    df = pd.read_parquet(os.path.join(root, "test.parquet"))

    if dataset_name == "dataset_a":
        query_paths, gallery_paths = [], []
        for pid, group in df.groupby("identity"):
            paths = group["image_path"].values.tolist()
            if len(paths) >= 2:
                query_paths.extend(paths[:2])
                gallery_paths.extend(paths[2:])
            else:
                gallery_paths.extend(paths)
        gallery_paths.extend(query_paths)  # standard ReID protocol
    else:
        query_df = df[df["split"] == "query"]
        gallery_df = df[df["split"] == "gallery"]
        query_paths = query_df["image_path"].tolist()
        gallery_paths = gallery_df["image_path"].tolist()

    return query_paths, gallery_paths


def predict(args):
    """Generate prediction CSV from a trained checkpoint.

    Loads the trained model, encodes query and gallery images, computes
    cosine similarity, and outputs ranked gallery indices per query.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model = TrainableModel(embedding_dim=args.embedding_dim).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {checkpoint.get('epoch', '?')})")

    # Load query/gallery paths
    query_paths, gallery_paths = load_query_gallery(args.dataset_root, args.dataset_name)
    print(f"Queries: {len(query_paths)}, Gallery: {len(gallery_paths)}")

    # Encode
    img_size = (args.image_size, args.image_size)
    query_emb = encode_images(model, ImageDataset(args.dataset_root, query_paths, img_size),
                              args.batch_size, args.num_workers, device)
    gallery_emb = encode_images(model, ImageDataset(args.dataset_root, gallery_paths, img_size),
                                args.batch_size, args.num_workers, device)

    # Compute rankings
    print("Computing rankings...")
    similarity = np.matmul(query_emb, gallery_emb.T)
    rankings = np.argsort(-similarity, axis=1)[:, :args.top_k]

    # Save predictions
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for q_idx in range(len(query_paths)):
        ranked_str = ",".join(str(x) for x in rankings[q_idx])
        rows.append({"query_index": q_idx, "ranked_gallery_indices": ranked_str})

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Predictions saved to: {args.output} ({len(rows)} queries)")


def main():
    parser = argparse.ArgumentParser(description="Object ReID Training Example")

    # Mode
    parser.add_argument("--predict", action="store_true", help="Generate predictions from a checkpoint")

    # Training args
    parser.add_argument("--data_root", type=str, default="./datasets/dataset_a", help="Dataset root for training")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--loss", type=str, default="arcface", choices=["arcface", "triplet"], help="Loss function")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Warmup epochs")
    parser.add_argument("--margin", type=float, default=0.3, help="Triplet loss margin")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")

    # Prediction args
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pth", help="Checkpoint path for prediction")
    parser.add_argument("--dataset_root", type=str, help="Dataset root for prediction")
    parser.add_argument("--dataset_name", type=str, choices=["dataset_a", "dataset_b"], help="Dataset name for prediction")
    parser.add_argument("--output", type=str, default="predictions/dataset_a.csv", help="Output CSV path for predictions")
    parser.add_argument("--top_k", type=int, default=50, help="Number of ranked results per query")

    # Shared args
    parser.add_argument("--embedding_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    if args.predict:
        if not args.dataset_root:
            parser.error("--dataset_root is required for prediction mode")
        if not args.dataset_name:
            parser.error("--dataset_name is required for prediction mode")
        predict(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
