#!/usr/bin/env python3
"""
COMP560 Object Re-Identification - CLIP Prediction & CSV Generator
"""

import argparse
import os
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
# Smart Dataset Loader (Finds images no matter what folder they are in)
# ============================================================================
class ImageDataset(Dataset):
    def __init__(self, root, image_paths, image_size=(224, 224)):
        self.root = root
        self.image_paths = []
        
        # Build a map of EVERY file in dataset_a so we don't miss test images
        print(f"Mapping test images in {root}...")
        self.file_map = {}
        for d, _, files in os.walk(root):
            for f in files:
                if f.endswith(('.jpg', '.jpeg', '.png')):
                    self.file_map[f] = os.path.join(d, f)
                    
        # Match requested paths to actual files
        for p in image_paths:
            filename = os.path.basename(p)
            if filename in self.file_map:
                self.image_paths.append(self.file_map[filename])
            else:
                self.image_paths.append(os.path.join(root, p))

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image), idx

# ============================================================================
# Minimal Model (Just enough to encode images)
# ============================================================================
class CLIPReID_Stage2(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.backbone = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch16")
        
    def encode(self, images):
        self.eval()
        with torch.no_grad():
            image_embeds = self.backbone(images).image_embeds
            return F.normalize(image_embeds, p=2, dim=-1)

# ============================================================================
# Encoding & Ranking Logic
# ============================================================================
def encode_images(model, dataset, batch_size, num_workers, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    emb_list, idx_list = [], []
    for images, indices in tqdm(loader, desc="Encoding Images"):
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
    return query_paths, gallery_paths

def main(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the Stage 2 Model
    model = CLIPReID_Stage2(embedding_dim=args.embedding_dim).to(device)
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # 2. Get Query and Gallery Paths
    query_paths, gallery_paths = load_query_gallery(args.data_root, args.dataset_name)
    print(f"Total Queries: {len(query_paths)} | Total Gallery: {len(gallery_paths)}")

    # 3. Encode all images (We can use a higher batch size here because no gradients are stored)
    img_size = (args.image_size, args.image_size)
    print("--- Encoding Query Images ---")
    query_emb = encode_images(model, ImageDataset(args.data_root, query_paths, img_size), 64, args.num_workers, device)
    
    print("--- Encoding Gallery Images ---")
    gallery_emb = encode_images(model, ImageDataset(args.data_root, gallery_paths, img_size), 64, args.num_workers, device)

    # 4. Compute similarity and rank
    print("Computing Rankings...")
    similarity = np.matmul(query_emb, gallery_emb.T)
    rankings = np.argsort(-similarity, axis=1)[:, :args.top_k]

    # 5. Save to CSV
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    rows = [{"query_index": q_idx, "ranked_gallery_indices": ",".join(str(x) for x in rankings[q_idx])} for q_idx in range(len(query_paths))]
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Predictions successfully saved to: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/stage2_best_model.pth")
    parser.add_argument("--data_root", type=str, default="./datasets/dataset_a")
    parser.add_argument("--dataset_name", type=str, default="dataset_a")
    parser.add_argument("--output", type=str, default="predictions_a.csv")
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()
    main(args)