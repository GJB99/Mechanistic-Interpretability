import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

def analyze_trajectory_characteristics(data):
    """Analyze the length and complexity of trajectories for each category"""
    trajectory_metrics = {}
    
    print("📏 Calculating trajectory metrics...")
    for idx, entry in enumerate(data):
        try:
            cat = entry['category']
            embeddings = np.array(entry['embeddings'])
            
            if cat not in trajectory_metrics:
                trajectory_metrics[cat] = {
                    'lengths': [],
                    'total_distance': 0.0,
                    'directness_ratio': [],
                    'layer_wise_changes': []
                }
            
            # Ensure embeddings is 2D (layers x features)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
                
            # Calculate total path length
            if embeddings.shape[0] > 1:
                diffs = np.diff(embeddings, axis=0)
                distances = np.linalg.norm(diffs, axis=1)
            else:
                distances = np.array([0.0])
                
            # Flatten any unexpected nested structure
            distances = distances.flatten().tolist()
            
            total_distance = sum(distances)
            
            # Calculate directness ratio
            if embeddings.shape[0] > 1:
                direct_distance = np.linalg.norm(embeddings[-1] - embeddings[0])
            else:
                direct_distance = 0.0
                
            directness = direct_distance / total_distance if total_distance != 0 else 0.0
            
            trajectory_metrics[cat]['lengths'].append(total_distance)
            trajectory_metrics[cat]['directness_ratio'].append(directness)
            trajectory_metrics[cat]['layer_wise_changes'].append(distances)
            trajectory_metrics[cat]['total_distance'] += total_distance
            
            if idx % 100 == 0:
                print(f"  Processed {idx+1}/{len(data)} entries")
                
        except Exception as e:
            print(f"🚨 Error processing entry {idx}: {str(e)}")
            print(f"Problematic entry data: {entry}")
            raise
    
    return trajectory_metrics

def analyze_starting_points(data):
    """Analyze the distribution and clustering of starting points"""
    # Extract first layer embeddings for each category
    category_starts = {}
    
    for entry in data:
        cat = entry['category']
        first_layer = np.array(entry['embeddings'])[0]  # First layer embedding
        
        if cat not in category_starts:
            category_starts[cat] = []
        category_starts[cat].append(first_layer)
    
    # Perform clustering analysis
    all_starts = np.vstack([np.mean(starts, axis=0) for starts in category_starts.values()])
    clustering = DBSCAN(eps=0.5, min_samples=2).fit(all_starts)
    
    try:
        silhouette = silhouette_score(all_starts, clustering.labels_)
    except:
        silhouette = 0  # Default if clustering fails
    
    return {
        'category_starts': category_starts,
        'cluster_labels': clustering.labels_,
        'silhouette_score': silhouette
    }

def analyze_endpoint_separation(data):
    """Study the relationship between endpoint separation and task performance"""
    
    # Extract final layer embeddings
    category_ends = {}
    for entry in data:
        cat = entry['category']
        final_layer = np.array(entry['embeddings'])[-1]
        
        if cat not in category_ends:
            category_ends[cat] = []
        category_ends[cat].append(final_layer)
    
    # Calculate inter-category distances
    distances = {}
    categories = list(category_ends.keys())
    
    for i, cat1 in enumerate(categories):
        for cat2 in categories[i+1:]:
            mean_end1 = np.mean(category_ends[cat1], axis=0)
            mean_end2 = np.mean(category_ends[cat2], axis=0)
            
            dist = np.linalg.norm(mean_end1 - mean_end2)
            distances[f"{cat1}_vs_{cat2}"] = dist
    
    return distances

def analyze_layer_transitions(data):
    """Analyze how representations change through layers"""
    layer_transitions = {}
    
    for entry in data:
        cat = entry['category']
        embeddings = np.array(entry['embeddings'])
        
        if cat not in layer_transitions:
            layer_transitions[cat] = []
        
        # Ensure 2D array and handle single-layer cases
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
            
        # Calculate layer changes with validation
        if embeddings.shape[0] > 1:
            layer_changes = np.diff(embeddings, axis=0)
            magnitudes = np.linalg.norm(layer_changes, axis=1).flatten().tolist()
        else:
            magnitudes = [0.0]  # Default for single-layer entries
            
        layer_transitions[cat].append(magnitudes)
    
    return layer_transitions

def create_enhanced_visualization(data, metrics, output_dir):
    """Create additional visualizations highlighting specific aspects"""
    
    # Plot 1: Trajectory Lengths
    plt.figure(figsize=(20, 12))
    plt.subplot(2, 2, 1)
    categories = list(metrics['trajectory_metrics'].keys())
    lengths = [metrics['trajectory_metrics'][cat]['lengths'] for cat in categories]
    plt.boxplot(lengths, tick_labels=categories, vert=False)
    plt.title("Trajectory Lengths by Category")
    plt.xlabel("Length")
    
    # Plot 2: Directness Ratios
    plt.subplot(2, 2, 2)
    directness = [metrics['trajectory_metrics'][cat]['directness_ratio'] for cat in categories]
    plt.boxplot(directness, tick_labels=categories, vert=False)
    plt.title("Path Directness (higher = more direct)")
    plt.xlabel("Directness Ratio")
    
    # Plot 3: Layer-wise Changes
    plt.subplot(2, 2, 3)
    for cat, transitions in metrics['layer_transitions'].items():
        mean_changes = np.mean(transitions, axis=0)
        plt.plot(mean_changes, label=cat)
    plt.title("Average Layer-wise Changes")
    plt.xlabel("Layer Transition")
    plt.ylabel("Magnitude of Change")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 4: Endpoint Distances Heatmap
    plt.subplot(2, 2, 4)
    distances_matrix = np.zeros((len(categories), len(categories)))
    for pair, dist in metrics['endpoint_distances'].items():
        cat1, cat2 = pair.split("_vs_")
        i, j = categories.index(cat1), categories.index(cat2)
        distances_matrix[i, j] = dist
        distances_matrix[j, i] = dist
    plt.imshow(distances_matrix)
    plt.colorbar(label='Distance')
    plt.title("Endpoint Distances Between Categories")
    plt.xticks(range(len(categories)), categories, rotation=45)
    plt.yticks(range(len(categories)), categories)
    
    plt.tight_layout()
    plt.savefig(output_dir / "enhanced_analysis.png", bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Load data
    results_dir = Path("results/24-01-2025-20-07-56")
    with open(results_dir / "cognitive_traces.json", 'r') as f:
        data = json.load(f)
    
    print("🔍 Analyzing cognitive traces...")
    
    # Run analyses
    metrics = {
        'trajectory_metrics': analyze_trajectory_characteristics(data),
        'starting_points': analyze_starting_points(data),
        'endpoint_distances': analyze_endpoint_separation(data),
        'layer_transitions': analyze_layer_transitions(data)
    }
    
    # Create visualizations
    print("📊 Creating enhanced visualizations...")
    create_enhanced_visualization(data, metrics, results_dir)
    
    # Save metrics
    print("💾 Saving analysis results...")
    try:
        with open(results_dir / "analysis_metrics.json", 'w') as f:
            serializable_metrics = {
                'trajectory_metrics': {},
                'starting_points': {
                    'cluster_labels': [int(x) for x in metrics['starting_points']['cluster_labels']],
                    'silhouette_score': float(metrics['starting_points']['silhouette_score'])
                },
                'endpoint_distances': {
                    k: float(v) for k, v in metrics['endpoint_distances'].items()
                },
                'layer_transitions': {
                    cat: [
                        [float(x) for x in transition_list]  # transition_list is already flattened
                        for transition_list in transitions
                    ]
                    for cat, transitions in metrics['layer_transitions'].items()
                }
            }
            
            # Add trajectory metrics with validation
            for cat in metrics['trajectory_metrics']:
                print(f"🔧 Processing category: {cat}")
                cat_metrics = metrics['trajectory_metrics'][cat]
                
                # Validate layer_wise_changes structure
                valid_changes = []
                for i, arr in enumerate(cat_metrics['layer_wise_changes']):
                    if isinstance(arr, list):
                        valid_changes.append([float(x) for x in arr])
                    else:
                        print(f"⚠️ Unexpected type in layer_wise_changes[{i}]: {type(arr)}")
                        valid_changes.append([float(arr)])
                
                serializable_metrics['trajectory_metrics'][cat] = {
                    'lengths': [float(x) for x in cat_metrics['lengths']],
                    'total_distance': float(cat_metrics['total_distance']),
                    'directness_ratio': [float(x) for x in cat_metrics['directness_ratio']],
                    'layer_wise_changes': valid_changes
                }
            
            json.dump(serializable_metrics, f, indent=2)
            
    except Exception as e:
        print("🔥 Critical error during serialization:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("💡 Check the last processed category and entry index above")
        raise

    print("✅ Analysis complete! Check the results directory for outputs.")

if __name__ == "__main__":
    main() 