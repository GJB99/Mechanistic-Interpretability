# Neural Pathway Analysis of Cognitive Tasks

This repository analyzes how different cognitive tasks manifest in the neural activations of the DeepSeek-R1 model through layer-wise activation patterns.

## Cognitive Task Taxonomy

The 15 cognitive categories and their focus areas:

| Category | Description |
|----------|-------------|
| **Vision** | Visual perception, spatial relationships, and object recognition |
| **Math** | Arithmetic operations, problem solving, and mathematical reasoning |
| **Language** | Linguistic creativity, translation, and semantic understanding |
| **Emotion** | Emotional recognition, social intelligence, and affective states |
| **Pattern Recognition** | Sequence completion, analogies, and logical patterns |
| **Logical Reasoning** | Deductive/inductive reasoning, syllogisms, and paradox resolution |
| **Spatial Reasoning** | Mental rotation, 3D visualization, and spatial transformations |
| **Creative Thinking** | Divergent thinking, metaphorical reasoning, and idea generation |
| **Social Intelligence** | Theory of mind, social norms, and interpersonal dynamics |
| **Working Memory** | Information maintenance and manipulation under load |
| **Executive Function** | Planning, task switching, and cognitive control |
| **Spatial Navigation** | Pathfinding, mental mapping, and orientation |
| **Temporal Processing** | Time estimation, event sequencing, and rhythm perception |
| **Decision Making** | Risk assessment, cost-benefit analysis, and uncertainty |
| **Motor Planning** | Action sequencing and kinesthetic imagination |

## Methodology

1. **Stimulus Presentation**: 200+ questions across 15 categories ([full list](questions.json))
2. **Activation Capture**: Layer-wise mean activations sampled every 4 layers
3. **Dimensionality Reduction**: UMAP projection to 2D space
4. **Visualization**: Color-coded trajectory plots showing layer progression

Some results: ![thought_trajectory_All_Categories](https://github.com/user-attachments/assets/3ba875c1-b5ae-470a-ab0a-854cf0436da5)
![enhanced_analysis](https://github.com/user-attachments/assets/15696c94-af62-4ac4-8fd3-8b6fae1a1be0)


## Key Features

- 15 cognitive categories with 10-15 questions each
- Layer-wise activation trajectories (cool → warm colors)
- Comparative analysis across cognitive domains
- Resource-efficient sampling strategy

## Usage
Install dependencies
pip install -r requirements.txt
Run analysis
python mechanistic_interpretability.py

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- DeepSeek-R1-Distill-Qwen-32B model access

See [requirements.txt](requirements.txt) for full dependency list.

This documentation structure:
1. Explains the cognitive taxonomy
2. Shows methodology
3. Provides visual examples
4. Includes setup/usage instructions
5. Maintains direct connection to the code's functionality
