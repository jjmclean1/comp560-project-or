#!/usr/bin/env python3
"""
COMP560 Object Re-Identification - CLIPReID Stage 2 (Mac Optimized)
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
# Dataset (With the 110k Deep Search Fix)
# ============================================================================
class ReIDTrainDataset(Dataset):
    def __init__(self, root: str, parquet_file: str = "or_dataset_a_train.parquet", image_size=(224, 224)):
        self.root = root
        
        parquet_path = os.path.join(root, parquet_file)
        if not os.path.exists(parquet_path):
            parquet_path = os.path.join(root, "..", parquet_file)
            
        df = pd.read_parquet(parquet_path)
        if "split" in df.columns:
            df = df[df["split"] == "train"]

        print("Building local image map (indexing 110k images)...")
        self.local_file_map = {}
        train_images_dir = os.path.join(self.root, "train_images")
        for root_dir, _, files in os.walk(train_images_dir):
            for f in files:
                if f.endswith(('.jpg', '.jpeg', '.png')):
                    self.local_file_map[f] = os.path.relpath(os.path.join(root_dir, f), self.root)

        def get_real_path(p):
            if pd.isna(p): return None
            filename = os.path.basename(p.strip())
            if filename in self.local_file_map:
                return self.local_file_map[filename]
            if os.path.exists(os.path.join(self.root, p.strip())):
                return p.strip()
            return None

        print("Mapping metadata to local files...")
        df["final_path"] = df["image_path"].apply(get_real_path)
        df = df[df["final_path"].notna()]
        
        print("Mapping metadata to local files...")
        df["final_path"] = df["image_path"].apply(get_real_path)
        df = df[df["final_path"].notna()]

        
        print(f"DEBUG: Using FULL dataset with {len(df)} valid images.")
        
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
        path = os.path.join(self.root, self.image_paths[idx])
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, self.labels[idx]

# ArcFace Loss (Stage 1 Token Initialized)
class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_dim, num_classes, s=30.0, m=0.50, init_weights=None):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        
        #use them tokens
        if init_weights is not None:
            self.weight.data = init_weights.clone()
        else:
            nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))

        one_hot = torch.zeros_like(cosine).scatter_(1, labels.unsqueeze(1), 1.0)
        target_logits = torch.cos(theta + self.m * one_hot)

        logits = target_logits * self.s
        return F.cross_entropy(logits, labels)

#unfrozen
class CLIPReID_Stage2(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.backbone = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
        
        for param in self.backbone.parameters():
            param.requires_grad = True

        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self):
        return self._embedding_dim

    def forward(self, images):
        image_embeds = self.backbone(images).image_embeds
        return F.normalize(image_embeds, p=2, dim=-1)

    def encode(self, images):
        self.eval()
        return self.forward(images)


def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = ReIDTrainDataset(args.data_root, image_size=(args.image_size, args.image_size))
    
    #smaller physical batch size
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True
    )

    model = CLIPReID_Stage2(embedding_dim=args.embedding_dim).to(device)

    # Load Stage 1
    print(f"Loading Stage 1 weights from {args.stage1_checkpoint}...")
    checkpoint = torch.load(args.stage1_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print("Stage 1 image backbone weights loaded.")

    #Extracting Stage 1 Text Tokens and give them to ArcFace
    stage1_tokens = checkpoint["model_state_dict"].get("identity_tokens.weight")
    
    if stage1_tokens is not None:
        print("SUCCESS: Initializing ArcFace weights using Stage 1 text tokens!")
        criterion = ArcFaceLoss(args.embedding_dim, dataset.num_classes, init_weights=stage1_tokens).to(device)
    else:
        print("WARNING: Stage 1 tokens not found. Using random ArcFace weights.")
        criterion = ArcFaceLoss(args.embedding_dim, dataset.num_classes).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1}, 
        {"params": criterion.parameters(), "lr": args.lr}
    ], weight_decay=args.weight_decay)

    #calculation accounts for gradient accumulation steps
    total_steps = (len(dataloader) // args.accumulate_steps) * args.epochs
    warmup_steps = (len(dataloader) // args.accumulate_steps) * args.warmup_epochs

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
        optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            embeddings = model(images)
            
            #divide loss by accumulation steps so gradients scale correctly
            loss = criterion(embeddings, labels) / args.accumulate_steps
            loss.backward()

            if (i + 1) % args.accumulate_steps == 0 or (i + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            display_loss = loss.item() * args.accumulate_steps
            epoch_loss += display_loss
            pbar.set_postfix(loss=f"{display_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = save_dir / "stage2_final_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'embedding_dim': args.embedding_dim,
                'num_classes': dataset.num_classes
            }, checkpoint_path)
            print(f"  Saved Stage 2 model to {checkpoint_path}")

def main():
    parser = argparse.ArgumentParser(description="CLIPReID Stage 2 Training (Mac Optimized)")
    parser.add_argument("--data_root", type=str, default="./datasets/dataset_a")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--stage1_checkpoint", type=str, default="./checkpoints/stage1_full_100k.pth")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=6) #Stage 2 takes longer, 6 is safe
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16) #LOW to protect memory
    parser.add_argument("--accumulate_steps", type=int, default=4) #batch of 64
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
    