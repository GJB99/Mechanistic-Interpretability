"""
This script creates a combined visualization of all cognitive traces from the latest results directory.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from umap import UMAP
import matplotlib.pyplot as plt

def find_latest_results():
    results_dir = Path("results")
    if not results_dir.exists():
        raise FileNotFoundError("No results directory found")
    
    # Get all timestamped directories
    dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not dirs:
        raise FileNotFoundError("No result runs found in results directory")
    
    # Sort by modification time
    latest_dir = max(dirs, key=lambda d: d.stat().st_mtime)
    return latest_dir

def load_cognitive_traces(results_dir):
    traces_file = results_dir / "cognitive_traces.json"
    if not traces_file.exists():
        raise FileNotFoundError(f"cognitive_traces.json not found in {results_dir}")
    
    with open(traces_file, "r") as f:
        data = json.load(f)
    
    # Convert embeddings back to numpy arrays
    for entry in data:
        entry["embeddings"] = np.array(entry["embeddings"])
    
    return data

def create_combined_visualization(data, output_dir):
    # Combine all embeddings across questions
    all_embeddings = np.concatenate([entry["embeddings"] for entry in data])
    
    # Reshape to 2D if needed (layers * questions × hidden_dim)
    if all_embeddings.ndim == 3:
        all_embeddings = all_embeddings.reshape(-1, all_embeddings.shape[-1])
    
    # Reduce dimensionality
    reducer = UMAP(n_components=2, random_state=42, n_jobs=-1)
    embeddings_2d = reducer.fit_transform(all_embeddings)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=np.linspace(0, 1, len(embeddings_2d)),
                         cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label="Layer Depth")
    plt.title("Combined Cognitive Trajectories\nLayer Progression: Cool → Warm")
    
    # Save visualization
    output_path = output_dir / "thought_trajectory_All_Categories.png"
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved combined visualization to: {output_path}")

if __name__ == "__main__":
    try:
        # Find latest results
        results_dir = find_latest_results()
        print(f"📂 Using latest results from: {results_dir}")
        
        # Load data
        traces_data = load_cognitive_traces(results_dir)
        print(f"📊 Loaded {len(traces_data)} question traces")
        
        # Create visualization
        create_combined_visualization(traces_data, results_dir)
        
    except Exception as e:
        print(f"❌ Error: {e}") 