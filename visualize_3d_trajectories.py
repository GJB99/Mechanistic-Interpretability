import json
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from umap import UMAP
import plotly.express as px

def load_traces(filepath):
    """Load the cognitive traces from JSON file"""
    print(f"📂 Loading traces from: {filepath}")
    with open(filepath, 'r') as f:
        return json.load(f)

def create_3d_visualization(data, output_dir):
    """Create interactive 3D visualization of cognitive trajectories"""
    print("🎨 Creating 3D visualization...")
    
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
    
    print(f"📊 Reducing dimensionality of {all_embeddings.shape} embeddings...")
    
    # Reduce dimensionality to 3D
    reducer = UMAP(
        n_components=3,
        random_state=42,
        n_jobs=-1,
        min_dist=0.1,
        n_neighbors=15
    )
    embeddings_3d = reducer.fit_transform(all_embeddings)
    
    # Create interactive 3D plot
    fig = go.Figure()
    
    # Track current position in the combined embeddings
    current_idx = 0
    
    # Get color palette
    colors = px.colors.qualitative.Set3
    
    print("🔍 Plotting trajectories for each category...")
    
    # Plot trajectories for each category
    for cat_idx, (cat, embeddings_list) in enumerate(categories.items()):
        # Calculate number of points for this category
        n_points = sum(emb.shape[0] for emb in embeddings_list)
        
        # Get this category's section of the UMAP projection
        cat_points = embeddings_3d[current_idx:current_idx + n_points]
        
        # Add trajectory line
        fig.add_trace(go.Scatter3d(
            x=cat_points[:, 0],
            y=cat_points[:, 1],
            z=cat_points[:, 2],
            mode='lines',
            name=f'{cat} (path)',
            line=dict(
                color=colors[cat_idx % len(colors)],
                width=2
            ),
            opacity=0.7
        ))
        
        # Add start point
        fig.add_trace(go.Scatter3d(
            x=[cat_points[0, 0]],
            y=[cat_points[0, 1]],
            z=[cat_points[0, 2]],
            mode='markers',
            name=f'{cat} (start)',
            marker=dict(
                size=8,
                symbol='circle',
                color=colors[cat_idx % len(colors)],
                line=dict(color='black', width=1)
            )
        ))
        
        # Add end point
        fig.add_trace(go.Scatter3d(
            x=[cat_points[-1, 0]],
            y=[cat_points[-1, 1]],
            z=[cat_points[-1, 2]],
            mode='markers',
            name=f'{cat} (end)',
            marker=dict(
                size=8,
                symbol='square',
                color=colors[cat_idx % len(colors)],
                line=dict(color='black', width=1)
            )
        ))
        
        current_idx += n_points
    
    # Update layout
    fig.update_layout(
        title='3D Cognitive Trajectories',
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        width=1200,
        height=800
    )
    
    # Save as HTML for interactivity
    output_path_html = output_dir / "cognitive_trajectories_3d.html"
    print(f"💾 Saving interactive HTML visualization...")
    fig.write_html(output_path_html)
    print(f"✅ Saved to: {output_path_html}")
    
    # Try to save static image if kaleido is available
    try:
        print(f"💾 Attempting to save static PNG (this might take a moment)...")
        output_path_png = output_dir / "cognitive_trajectories_3d.png"
        fig.write_image(output_path_png)
        print(f"✅ Saved to: {output_path_png}")
    except Exception as e:
        print(f"⚠️ Could not save PNG: {str(e)}")
        print("💡 To enable PNG export, install kaleido: pip install -U kaleido")

if __name__ == "__main__":
    results_dir = Path("results/24-01-2025-20-07-56")
    traces_file = results_dir / "cognitive_traces.json"
    
    traces_data = load_traces(traces_file)
    print(f"📊 Loaded {len(traces_data)} traces")
    
    create_3d_visualization(traces_data, results_dir)