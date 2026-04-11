#!/usr/bin/env python3
"""
COMP560 Object Re-Identification Evaluation Script

Evaluates student-submitted prediction files against ground truth.
Students submit a CSV file with ranked gallery indices for each query.

Submission format (CSV):
    query_index,ranked_gallery_indices
    0,"45,12,78,3,99,..."
    1,"102,5,67,23,11,..."
    ...

- query_index: integer index of the query image (0-based, matching test.parquet order for query split)
- ranked_gallery_indices: comma-separated gallery indices sorted by similarity (most similar first)
  At least top-50 indices should be provided.

Usage:
    python evaluate.py --student_id <your_id> --prediction <rankings.csv>
    python evaluate.py --student_id <your_id> --prediction <rankings.csv> --datasets dataset_a dataset_b
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_cmc_map(
    predictions: np.ndarray,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_camids: np.ndarray,
    gallery_camids: np.ndarray,
    max_rank: int = 50,
    exclude_same_camera: bool = True,
) -> Tuple[np.ndarray, float, float]:
    """Compute CMC curve and mAP following standard ReID evaluation protocol."""
    num_query = predictions.shape[0]

    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_query = 0

    for q_idx in range(num_query):
        q_pid = query_pids[q_idx]
        q_camid = query_camids[q_idx]

        # Get gallery items in predicted order
        order = predictions[q_idx]
        g_pids = gallery_pids[order]
        g_camids = gallery_camids[order]

        if exclude_same_camera:
            keep = ~((g_pids == q_pid) & (g_camids == q_camid))
        else:
            keep = np.ones(len(g_pids), dtype=bool)

        g_pids = g_pids[keep]
        matches = (g_pids == q_pid).astype(np.int32)

        if matches.sum() == 0:
            continue

        num_valid_query += 1

        cmc = matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])

        num_rel = matches.sum()
        tmp_cmc = matches.cumsum()
        precision_at_k = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        precision_at_k = np.asarray(precision_at_k) * matches
        AP = precision_at_k.sum() / num_rel
        all_AP.append(AP)

        pos_idx = np.where(matches == 1)[0]
        if len(pos_idx) > 0:
            max_pos_idx = np.max(pos_idx)
            inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
            all_INP.append(inp)

    if num_valid_query == 0:
        return np.zeros(max_rank), 0.0, 0.0

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    cmc = all_cmc.sum(axis=0) / num_valid_query

    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP) if all_INP else 0.0

    return cmc, mAP, mINP


# ============================================================================
# Dataset Loading
# ============================================================================

def load_dataset_a_gt(root: str):
    """Load dataset_a ground truth: test.parquet with dynamic query/gallery split."""
    df = pd.read_parquet(os.path.join(root, "test.parquet"))

    query_rows = []
    gallery_rows = []

    for pid, group in df.groupby("identity"):
        rows = group.to_dict("records")
        if len(rows) >= 2:
            query_rows.extend(rows[:2])
            gallery_rows.extend(rows[2:])
        else:
            gallery_rows.extend(rows)

    # Also add query images to gallery (standard ReID protocol)
    gallery_rows.extend(query_rows)

    query_pids = np.array([r["identity"] for r in query_rows])
    query_camids = np.array([r["camera_id"] for r in query_rows])
    gallery_pids = np.array([r["identity"] for r in gallery_rows])
    gallery_camids = np.array([r["camera_id"] for r in gallery_rows])

    return query_pids, query_camids, gallery_pids, gallery_camids, len(query_rows), len(gallery_rows)


def load_dataset_b_gt(root: str):
    """Load dataset_b ground truth: test.parquet with explicit query/gallery."""
    df = pd.read_parquet(os.path.join(root, "test.parquet"))
    query_df = df[df["split"] == "query"]
    gallery_df = df[df["split"] == "gallery"]

    return (
        query_df["identity"].values,
        query_df["camera_id"].values,
        gallery_df["identity"].values,
        gallery_df["camera_id"].values,
        len(query_df),
        len(gallery_df),
    )


# ============================================================================
# Main Evaluation
# ============================================================================

def evaluate_dataset(
    prediction_path: str,
    dataset_root: str,
    dataset_name: str,
    k_values: List[int] = [1, 5, 10, 20],
) -> Dict:
    """Evaluate predictions against ground truth."""

    # Load ground truth
    if dataset_name == "dataset_a":
        query_pids, query_camids, gallery_pids, gallery_camids, n_q, n_g = load_dataset_a_gt(dataset_root)
        exclude_same_camera = False
    else:
        query_pids, query_camids, gallery_pids, gallery_camids, n_q, n_g = load_dataset_b_gt(dataset_root)
        exclude_same_camera = True

    # Load predictions
    pred_df = pd.read_csv(prediction_path)
    required_cols = {"query_index", "ranked_gallery_indices"}
    if not required_cols.issubset(pred_df.columns):
        raise ValueError(f"Prediction CSV must have columns: {required_cols}. Got: {set(pred_df.columns)}")

    # Parse ranked indices
    max_rank = max(k_values) + 10
    predictions = np.zeros((n_q, min(max_rank, n_g)), dtype=np.int64)

    for _, row in pred_df.iterrows():
        q_idx = int(row["query_index"])
        if q_idx >= n_q:
            continue
        indices_str = str(row["ranked_gallery_indices"]).strip('"').strip("'")
        indices = [int(x.strip()) for x in indices_str.split(",") if x.strip()]
        length = min(len(indices), predictions.shape[1])
        predictions[q_idx, :length] = indices[:length]

    # Check coverage
    predicted_queries = set(pred_df["query_index"].astype(int))
    missing = n_q - len(predicted_queries.intersection(range(n_q)))
    if missing > 0:
        print(f"  WARNING: {missing}/{n_q} queries have no prediction")

    # Compute metrics
    cmc, mAP, mINP = compute_cmc_map(
        predictions, query_pids, gallery_pids, query_camids, gallery_camids,
        max_rank=max_rank, exclude_same_camera=exclude_same_camera,
    )

    rank_metrics = {}
    for k in k_values:
        if k <= len(cmc):
            rank_metrics[f"Rank-{k}"] = cmc[k - 1] * 100
        else:
            rank_metrics[f"Rank-{k}"] = cmc[-1] * 100 if len(cmc) > 0 else 0.0

    results = {
        "performance": {
            **rank_metrics,
            "mAP": mAP * 100,
            "mINP": mINP * 100,
            "combined": (mAP + cmc[0]) / 2 * 100,
        },
        "submission_info": {
            "num_queries": n_q,
            "num_gallery": n_g,
            "num_predicted_queries": len(predicted_queries.intersection(range(n_q))),
            "num_missing_queries": missing,
        },
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="COMP560 Object Re-Identification Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--student_id", type=str, required=True, help="Your student ID")
    parser.add_argument(
        "--prediction", type=str, required=True,
        help="Path to prediction CSV (or directory with dataset_a.csv, dataset_b.csv)",
    )
    parser.add_argument("--datasets_root", type=str, default="./datasets", help="Root directory containing datasets")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument(
        "--datasets", type=str, nargs="+",
        default=["dataset_a", "dataset_b"],
        choices=["dataset_a", "dataset_b"],
        help="Datasets to evaluate on",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("COMP560 Object Re-Identification Evaluation")
    print("=" * 60)
    print(f"Student ID: {args.student_id}")
    print(f"Prediction: {args.prediction}")
    print(f"Datasets: {args.datasets}")
    print("=" * 60)

    all_results = {
        "student_id": args.student_id,
        "timestamp": timestamp,
        "prediction_path": args.prediction,
        "datasets": {},
    }

    for dataset_name in args.datasets:
        print(f"\n{'=' * 60}")
        print(f"Evaluating on {dataset_name}")
        print("=" * 60)

        dataset_root = os.path.join(args.datasets_root, dataset_name)
        if not os.path.exists(os.path.join(dataset_root, "test.parquet")):
            print(f"  ERROR: Ground truth not found at {dataset_root}")
            all_results["datasets"][dataset_name] = {"error": "ground truth not found"}
            continue

        if os.path.isdir(args.prediction):
            pred_path = os.path.join(args.prediction, f"{dataset_name}.csv")
        else:
            pred_path = args.prediction

        if not os.path.exists(pred_path):
            print(f"  ERROR: Prediction file not found at {pred_path}")
            all_results["datasets"][dataset_name] = {"error": "prediction file not found"}
            continue

        try:
            results = evaluate_dataset(pred_path, dataset_root, dataset_name)
            all_results["datasets"][dataset_name] = results

            print(f"\nPerformance Metrics:")
            for metric, value in results["performance"].items():
                print(f"  {metric}: {value:.2f}%")

            print(f"\nSubmission Info:")
            for key, value in results["submission_info"].items():
                print(f"  {key}: {value}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results["datasets"][dataset_name] = {"error": str(e)}

    output_file = output_dir / f"{args.student_id}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    summary_file = output_dir / f"{args.student_id}_{timestamp}_summary.csv"
    with open(summary_file, "w") as f:
        f.write("dataset,Rank-1,Rank-5,Rank-10,Rank-20,mAP,mINP,combined\n")
        for dataset_name, results in all_results["datasets"].items():
            if "error" not in results:
                perf = results["performance"]
                f.write(f"{dataset_name},{perf.get('Rank-1', 0):.2f},{perf.get('Rank-5', 0):.2f},"
                       f"{perf.get('Rank-10', 0):.2f},{perf.get('Rank-20', 0):.2f},"
                       f"{perf.get('mAP', 0):.2f},{perf.get('mINP', 0):.2f},{perf.get('combined', 0):.2f}\n")
    print(f"Summary saved to: {summary_file}")

    return all_results


if __name__ == "__main__":
    main()
