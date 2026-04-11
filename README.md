# COMP560 Project: Object Re-Identification

## Task

Build a system that retrieves the most similar objects from a gallery given a query image. For each query, produce a ranked list of gallery indices.

## Datasets

| Dataset | Train | Test (Query/Gallery) | Identities | Image Size |
|---------|-------|---------------------|------------|------------|
| dataset_a | ~110K | ~31K (split dynamically) | ~10K | 224x224 |
| dataset_b | ~38K | ~13K (1.7K query / 11.6K gallery) | ~776 | 224x224 |

Data format (Parquet):
- `train.parquet`: training images (image_path, split, identity, camera_id)
- `test.parquet`: test images (image_path, split, identity, camera_id)

**dataset_a**: No real camera info. Query/gallery split constructed dynamically (2 per identity for query, rest for gallery, query also added to gallery).

**dataset_b**: Has camera info. Explicit query/gallery split (split="query" or "gallery"). Same-camera matches excluded during evaluation.

## Submission Format

Submit a CSV file per dataset with columns:

```csv
query_index,ranked_gallery_indices
0,"45,12,78,3,99,..."
1,"102,5,67,23,11,..."
...
```

- `query_index`: 0-based query index
- `ranked_gallery_indices`: comma-separated gallery indices sorted by similarity (most similar first), at least top-50

## Evaluation

```bash
# Evaluate on dataset_a
python evaluate.py --student_id YOUR_ID --prediction predictions/dataset_a.csv --datasets dataset_a

# Evaluate on both (pass a directory)
python evaluate.py --student_id YOUR_ID --prediction predictions/ --datasets dataset_a dataset_b
```

## Baseline

Generate baseline predictions using pretrained ResNet50:

```bash
python models/resnet_baseline.py --dataset_root ./datasets/dataset_a --dataset_name dataset_a --output predictions/dataset_a.csv
python models/resnet_baseline.py --dataset_root ./datasets/dataset_b --dataset_name dataset_b --output predictions/dataset_b.csv
```

## Training Example

Train a ResNet50 model with ArcFace or triplet loss:

```bash
python train_example.py --data_root ./datasets/dataset_a --loss arcface --epochs 20
python train_example.py --data_root ./datasets/dataset_b --loss triplet --epochs 20
```

Generate predictions from a trained checkpoint:

```bash
python train_example.py --predict --checkpoint ./checkpoints/best_model.pth --dataset_root ./datasets/dataset_a --dataset_name dataset_a --output predictions/dataset_a.csv
python train_example.py --predict --checkpoint ./checkpoints/best_model.pth --dataset_root ./datasets/dataset_b --dataset_name dataset_b --output predictions/dataset_b.csv
```

## Metrics

- **Rank-1, Rank-5, Rank-10, Rank-20**: Probability of correct match within top-K
- **mAP**: Mean Average Precision
- **mINP**: Mean Inverse Negative Penalty

## Grading

- 40% Performance (Rank-K, mAP metrics)
- 30% Efficiency (model design, embedding dimension)
- 30% Report

## Directory Structure

```
project-or/
├── datasets/
│   ├── dataset_a/
│   │   ├── images/          # Object images
│   │   ├── train.parquet
│   │   └── test.parquet
│   └── dataset_b/
│       ├── images/
│       │   ├── train/
│       │   ├── query/
│       │   └── gallery/
│       ├── train.parquet
│       └── test.parquet
├── models/
│   └── resnet_baseline.py   # Baseline prediction generator
├── evaluate.py              # Evaluation script
├── train_example.py         # Training example
└── results/                 # Output directory
```
