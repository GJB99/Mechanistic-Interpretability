import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.linear_model import LinearRegression

def load_traces(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def create_trajectory_visualization(data, output_dir):
    # Group embeddings by category
    categories = {}
    for entry in data:
        cat = entry['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(np.array(entry['embeddings']))
    
    # Combine all embeddings for UMAP
    all_embeddings = np.concatenate([np.array(entry['embeddings']) for entry in data])
    if all_embeddings.ndim == 3:
        all_embeddings = all_embeddings.reshape(-1, all_embeddings.shape[-1])
    
    # Reduce dimensionality
    reducer = UMAP(n_components=2, random_state=42, n_jobs=-1)
    embeddings_2d = reducer.fit_transform(all_embeddings)
    
    # Create plot
    plt.figure(figsize=(15, 10))
    
    # Track current position in the combined embeddings
    current_idx = 0
    
    # Color map for categories
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
    
    for (cat, embeddings_list), color in zip(categories.items(), colors):
        # Calculate number of points for this category
        n_points = sum(emb.shape[0] for emb in embeddings_list)
        
        # Get this category's section of the UMAP projection
        cat_points = embeddings_2d[current_idx:current_idx + n_points]
        
        # Create sequence for x-axis (layer numbers)
        x = np.arange(len(cat_points)).reshape(-1, 1)
        
        # Fit regression lines
        reg_x = LinearRegression().fit(x, cat_points[:, 0])
        reg_y = LinearRegression().fit(x, cat_points[:, 1])
        
        # Plot points and regression line
        plt.scatter(cat_points[:, 0], cat_points[:, 1], 
                   c=[color], alpha=0.3, label=cat)
        
        # Plot regression line
        line_x = np.array([0, len(x)]).reshape(-1, 1)
        plt.plot(reg_x.predict(line_x), reg_y.predict(line_x), 
                color=color, linewidth=2)
        
        # Mark start and end points
        start_point = cat_points[0]
        end_point = cat_points[-1]
        plt.scatter(start_point[0], start_point[1], color=color, s=100, marker='o', 
                   edgecolor='black', linewidth=1.5, label=f'{cat} (start)')
        plt.scatter(end_point[0], end_point[1], color=color, s=100, marker='s', 
                   edgecolor='black', linewidth=1.5, label=f'{cat} (end)')
        
        current_idx += n_points
    
    plt.title("Cognitive Category Trajectories\nwith Layer Progression Lines")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save visualization
    output_path = output_dir / "thought_trajectory_Categories_with_Lines.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved visualization to: {output_path}")

if __name__ == "__main__":
    results_dir = Path("results/24-01-2025-20-07-56")
    traces_file = results_dir / "cognitive_traces.json"
    
    print(f"📂 Loading traces from: {traces_file}")
    traces_data = load_traces(traces_file)
    print(f"📊 Loaded {len(traces_data)} traces")
    
    create_trajectory_visualization(traces_data, results_dir)