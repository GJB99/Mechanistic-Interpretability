"""
This script attempts to analyze the internal neural pathways of the Deepseek model (https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)
during different types of reasoning tasks. The goal is to understand which parts
of the network are activated for different cognitive tasks.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
import pathlib
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ModelPathwayAnalyzer:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"):
        """Initialize the pathway analyzer with DeepSeek model."""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.activation_cache = {}
        self.attention_patterns = {}
        self.register_hooks()
        
    def register_hooks(self):
        """Register forward hooks to capture activations at different layers."""
        def hook_fn(module, input, output, layer_name):
            self.activation_cache[layer_name] = output.detach()
        
        # Register hooks for transformer layers
        for name, module in self.model.named_modules():
            if "layers" in name:  # This catches the transformer layers
                if "attention" in name:
                    module.register_forward_hook(
                        lambda m, i, o, name=name: hook_fn(m, i, o, f"attention_{name}")
                    )
                elif "mlp" in name:
                    module.register_forward_hook(
                        lambda m, i, o, name=name: hook_fn(m, i, o, f"mlp_{name}")
                    )
    
    def analyze_pathway(self, question: str, category: str) -> Dict:
        """Analyze the neural pathway for a given question."""
        # Clear previous activations
        self.activation_cache.clear()
        
        # Format input using chat template
        messages = [{"role": "user", "content": question}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize and run through model
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Collect activation patterns
        pathway_data = {
            'layer_activations': {
                layer: activation.mean().item() 
                for layer, activation in self.activation_cache.items()
            },
            'attention_patterns': {
                layer: activation.mean(dim=(0, 1)).cpu().numpy().tolist()
                for layer, activation in self.activation_cache.items()
                if 'attention' in layer
            },
            'category': category,
            'question': question
        }
        
        return pathway_data
    
    def compare_pathways(self, category1: str, category2: str) -> Dict:
        """Compare neural pathways between two categories of questions."""
        cat1_activations = [data['layer_activations'] 
                          for data in self.pathway_analyses[category1]]
        cat2_activations = [data['layer_activations'] 
                          for data in self.pathway_analyses[category2]]
        
        comparison = {
            'activation_differences': {},
            'attention_pattern_differences': {}
        }
        
        # Compare average activations
        for layer in cat1_activations[0].keys():
            cat1_mean = np.mean([act[layer] for act in cat1_activations])
            cat2_mean = np.mean([act[layer] for act in cat2_activations])
            comparison['activation_differences'][layer] = cat1_mean - cat2_mean
            
        return comparison
    
    def visualize_pathway(self, pathway_data: Dict):
        """Visualize the neural pathway analysis."""
        plt.figure(figsize=(15, 10))
        
        # Plot activation patterns
        plt.subplot(2, 1, 1)
        activations = list(pathway_data['layer_activations'].values())
        layer_names = list(pathway_data['layer_activations'].keys())
        sns.barplot(x=list(range(len(activations))), y=activations)
        plt.xticks(range(len(layer_names)), layer_names, rotation=45)
        plt.title(f"Layer Activations for Category: {pathway_data['category']}")
        
        # Plot attention patterns if available
        if pathway_data['attention_patterns']:
            plt.subplot(2, 1, 2)
            attention_matrix = np.array(list(pathway_data['attention_patterns'].values()))
            sns.heatmap(attention_matrix, cmap='viridis')
            plt.title("Attention Patterns Across Layers")
        
        plt.tight_layout()
        plt.show()

def load_questions() -> Dict[str, List[str]]:
    """Load questions from existing results"""
    results_dir = pathlib.Path("results")
    if not results_dir.exists():
        return {}
        
    # Get most recent results
    run_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        return {}
        
    last_run_dir = max(run_dirs)
    response_file = last_run_dir / "responses_intermediate.json"
    
    try:
        with open(response_file, 'r') as f:
            data = json.load(f)
            return {category: list(qa_pairs.keys()) 
                   for category, qa_pairs in data.items()}
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def main():
    # Initialize analyzer
    analyzer = ModelPathwayAnalyzer()
    
    # Define some test categories and questions if no previous questions exist
    default_questions = {
        "logical_reasoning": [
            "If all A are B, and all B are C, what can we conclude about A and C?",
            "What is the logical fallacy in assuming correlation implies causation?"
        ],
        "mathematical_reasoning": [
            "How would you solve a quadratic equation step by step?",
            "Explain the concept of mathematical induction."
        ],
        "creative_thinking": [
            "How would you design a city on Mars?",
            "What are some unique solutions to reduce plastic waste?"
        ]
    }
    
    # Load questions from previous runs or use defaults
    questions = load_questions()
    if not questions:
        questions = default_questions
    
    # Analyze pathways for each category
    pathway_analyses = {}
    for category, category_questions in questions.items():
        pathway_analyses[category] = []
        print(f"Analyzing {category} questions...")
        for question in category_questions:
            pathway_data = analyzer.analyze_pathway(question, category)
            pathway_analyses[category].append(pathway_data)
            
            # Visualize each pathway as we go
            analyzer.visualize_pathway(pathway_data)
    
    # Compare pathways between categories
    categories = list(questions.keys())
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            cat1, cat2 = categories[i], categories[j]
            print(f"\nComparing {cat1} vs {cat2}:")
            comparison = analyzer.compare_pathways(cat1, cat2)
            print(json.dumps(comparison, indent=2))
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = pathlib.Path("pathway_results") / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / "pathway_analyses.json", 'w') as f:
        json.dump(pathway_analyses, f, indent=2)
    
    print(f"\nResults saved to {results_dir}")

if __name__ == "__main__":
    main()

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", device_map="auto")

# Create the pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,  # Adjust based on your needs
    do_sample=True,      # Enable sampling
    temperature=0.6,     # Recommended between 0.5-0.7 per docs
    top_p=0.95          # Default value from docs
)

# Format message using chat template
messages = [{"role": "user", "content": "Who are you?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Generate response
response = pipe(prompt)

# Extract generated text
generated_text = response[0]['generated_text']
print(generated_text)