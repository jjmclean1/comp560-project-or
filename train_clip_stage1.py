#!/usr/bin/env python3
"""
COMP560 Object Re-Identification - CLIPReID Stage 1
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection

# ============================================================================
# Dataset (Kept exactly as provided by the professor)
# ============================================================================
class ReIDTrainDataset(Dataset):
    """Training dataset for ReID loaded from Parquet metadata."""

    def __init__(self, root: str, parquet_file: str = "or_dataset_a_train.parquet", image_size=(224, 224)):
        self.root = root
        
        # 1. Try to find the parquet file (checks root, then checks one level up)
        parquet_path = os.path.join(root, parquet_file)
        if not os.path.exists(parquet_path):
            parquet_path = os.path.join(root, "..", parquet_file)
            
        df = pd.read_parquet(parquet_path)

        if "split" in df.columns:
            df = df[df["split"] == "train"]

        print("Scanning for missing images...")
        df["image_path"] = df["image_path"].str.strip()
        
        # This function finds the file whether your folder is named 'images' or 'train_images'
        def get_real_path(p):
            # Try original (images/...)
            path1 = os.path.join(self.root, p)
            if os.path.exists(path1): return p
            
            # Try renamed (train_images/...)
            path2 = os.path.join(self.root, p.replace("images/", "train_images/"))
            if os.path.exists(path2): return p.replace("images/", "train_images/")
            
            # Try direct (stripping the images/ prefix if root is already inside train_images)
            path3 = os.path.join(self.root, p.split("/", 1)[-1])
            if os.path.exists(path3): return p.split("/", 1)[-1]
            
            return None

        # Apply the path finder
        df["final_path"] = df["image_path"].apply(get_real_path)
        df = df[df["final_path"].notna()]
        
        print(f"DEBUG: Kept {len(df)} valid images.")
        
        # Essential: These must be defined even if count is 0 to avoid AttributeErrors
        self.image_paths = df["final_path"].tolist()
        unique_ids = sorted(df["identity"].unique())
        self.id_to_label = {pid: i for i, pid in enumerate(unique_ids)}
        self.num_classes = len(unique_ids)
        self.labels = [self.id_to_label[pid] for pid in df["identity"].values]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Uses the final_path we found during the scan
        path = os.path.join(self.root, self.image_paths[idx])
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, self.labels[idx]



# ============================================================================
# Model: CLIPReID Stage 1
# ============================================================================
class CLIPReID_Stage1(nn.Module):
    def __init__(self, num_classes, embedding_dim=512):
        super().__init__()
        # Load the off-the-shelf CLIP Vision Encoder
        self.backbone = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
        
        # STAGE 1 MAGIC: Freeze the entire image encoder
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Create a learnable token for each identity
        self.identity_tokens = nn.Embedding(num_classes, embedding_dim)
        
        # Learnable temperature scaling (standard in CLIP)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self):
        return self._embedding_dim

    def forward(self, images, labels=None):
        # 1. Get image embeddings from the frozen backbone
        image_embeds = self.backbone(images).image_embeds
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)

        # If training, calculate similarity against all identity tokens
        if self.training and labels is not None:
            text_embeds = self.identity_tokens.weight
            text_embeds = F.normalize(text_embeds, p=2, dim=-1)
            
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_embeds @ text_embeds.t()
            return logits
        
        return image_embeds

    def encode(self, images):
        # Used during evaluation/prediction to just get the image vector
        self.eval()
        return self.forward(images)

# ============================================================================
# Training Loop
# ============================================================================
def train(args):
    # Auto-detect Mac Apple Silicon (MPS), CUDA, or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dataset = ReIDTrainDataset(args.data_root, image_size=(args.image_size, args.image_size))
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )

    print(f"Training set: {len(dataset)} images, {dataset.num_classes} identities")

    model = CLIPReID_Stage1(num_classes=dataset.num_classes, embedding_dim=args.embedding_dim).to(device)

    # For Stage 1, we use basic CrossEntropyLoss (matching images to their token)
    criterion = nn.CrossEntropyLoss()

    # OPTIMIZER: ONLY train the identity tokens and the logit scale. The backbone is frozen!
    optimizer = torch.optim.AdamW([
        {"params": model.identity_tokens.parameters(), "lr": args.lr},
        {"params": [model.logit_scale], "lr": args.lr}
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

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images, labels)
            loss = criterion(logits, labels)

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
                'num_classes': dataset.num_classes # Save this so we can load it later!
            }, save_dir / "best_model.pth")
            print(f"  Saved best model (loss={avg_loss:.4f})")

# ============================================================================
# Prediction Generation (Unchanged)
# ============================================================================
class ImageDataset(Dataset):
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
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
        gallery_paths.extend(query_paths) 
    else:
        query_df = df[df["split"] == "query"]
        gallery_df = df[df["split"] == "gallery"]
        query_paths = query_df["image_path"].tolist()
        gallery_paths = gallery_df["image_path"].tolist()
    return query_paths, gallery_paths

def predict(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    num_classes = checkpoint.get('num_classes', 10000) # Fallback if missing
    
    model = CLIPReID_Stage1(num_classes=num_classes, embedding_dim=args.embedding_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    query_paths, gallery_paths = load_query_gallery(args.dataset_root, args.dataset_name)
    img_size = (args.image_size, args.image_size)
    query_emb = encode_images(model, ImageDataset(args.dataset_root, query_paths, img_size), args.batch_size, args.num_workers, device)
    gallery_emb = encode_images(model, ImageDataset(args.dataset_root, gallery_paths, img_size), args.batch_size, args.num_workers, device)

    print("Computing rankings...")
    similarity = np.matmul(query_emb, gallery_emb.T)
    rankings = np.argsort(-similarity, axis=1)[:, :args.top_k]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for q_idx in range(len(query_paths)):
        ranked_str = ",".join(str(x) for x in rankings[q_idx])
        rows.append({"query_index": q_idx, "ranked_gallery_indices": ranked_str})

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Predictions saved to: {args.output}")

def main():
    parser = argparse.ArgumentParser(description="CLIPReID Stage 1 Training")
    parser.add_argument("--predict", action="store_true", help="Generate predictions")
    parser.add_argument("--data_root", type=str, default="./datasets/dataset_a")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pth")
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--dataset_name", type=str, choices=["dataset_a", "dataset_b"])
    parser.add_argument("--output", type=str, default="predictions/dataset_a.csv")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()

    if args.predict:
        predict(args)
    else:
        train(args)

if __name__ == "__main__":
    main()