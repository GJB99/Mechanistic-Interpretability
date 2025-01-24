"""
This script attempts to analyze the internal neural pathways of a Deepseek model 
(https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen1.5B)
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
from umap import UMAP
import re

class ModelPathwayAnalyzer:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        print(f"\n🔧 Initializing analyzer with model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.activation_cache = []
        self.register_hooks()
        print(f"✅ Model loaded on device: {self.model.device}")
        
    def register_hooks(self):
        """Register hooks to capture layer activations efficiently"""
        print("🔌 Registering activation hooks...")
        
        def create_hook(layer_idx):
            def hook_fn(module, inputs, outputs):
                if isinstance(outputs, tuple):
                    activation = outputs[0]
                else:
                    activation = outputs
                self.activation_cache[layer_idx] = activation.mean(dim=1).detach().cpu()
            return hook_fn

        self.selected_layers = []
        layer_counter = 0
        
        # Only hook the main transformer layers (model.layers.0, model.layers.1, etc.)
        for name, module in self.model.named_modules():
            if re.match(r"model\.layers\.\d+$", name):  # Match exact layer modules
                print(f"🔗 Hooking layer {layer_counter}: {name}")
                module.register_forward_hook(create_hook(layer_counter))
                self.selected_layers.append(name)
                layer_counter += 1
        
        # Initialize cache with exact layer count
        self.activation_cache = [None] * layer_counter
        print(f"📡 Registered hooks on {layer_counter} main transformer layers")

    def get_thought_embeddings(self, question: str) -> np.ndarray:
        """Get layer-wise activations during response generation"""
        self.activation_cache = [None] * len(self.activation_cache)
        
        messages = [{"role": "user", "content": question}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate response while capturing activations
        with torch.no_grad():
            self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        if any(x is None for x in self.activation_cache):
            raise ValueError(f"Missing activations in {self.activation_cache.count(None)} layers")
            
        return torch.stack(self.activation_cache).numpy()

    def visualize_thought_space(self, embeddings: np.ndarray, category: str, output_dir: pathlib.Path):
        """Visualize activations using UMAP"""
        print(f"🎨 Generating visualization for {category}...")
        
        # Remove batch dimension and layer-wise concatenation
        if embeddings.ndim == 3:
            embeddings = embeddings.squeeze(1)  # Shape: (layers, hidden_dim)
        
        # Reduce dimensionality
        reducer = UMAP(
            n_components=2, 
            random_state=42,
            n_jobs=-1,
            transform_seed=42
        )
        embeddings_2d = reducer.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                   c=np.linspace(0, 1, len(embeddings_2d)),
                   cmap='viridis', alpha=0.7)
        plt.title(f"Thought Trajectory: {category}\nLayer Progression: Cool → Warm")
        plt.colorbar(label="Layer Depth")
        filename = f"thought_trajectory_{category}.png"
        plt.savefig(output_dir / filename)
        plt.close()
        print(f"💾 Saved visualization: {output_dir / filename}")

def load_questions() -> Dict[str, List[str]]:
    """Load questions from JSON file"""
    print("\n⏳ Loading questions.json...")
    questions_file = pathlib.Path("questions.json")
    
    if not questions_file.exists():
        raise FileNotFoundError(f"Questions file not found at {questions_file}")
    
    with open(questions_file, 'r') as f:
        data = json.load(f)
    
    # Validate structure
    if not isinstance(data, dict) or not all(isinstance(v, list) for v in data.values()):
        raise ValueError("Invalid questions.json format. Expected {category: [questions]}")
    
    total_questions = sum(len(v) for v in data.values())
    print(f"✅ Loaded {len(data)} categories with {total_questions} total questions")
    return data

def main():
    print("🚀 Starting neural pathway analysis")
    
    # Create timestamped results directory
    results_dir = pathlib.Path("results") / datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {results_dir}")
    
    analyzer = ModelPathwayAnalyzer()
    
    try:
        questions = load_questions()
    except Exception as e:
        print(f"❌ Error loading questions: {e}")
        return
    
    all_embeddings = []
    categories = []
    total_processed = 0
    category_count = len(questions.items())
    
    print("\n🔍 Beginning category processing...")
    for cat_idx, (category, category_questions) in enumerate(questions.items(), 1):
        print(f"\n📂 Processing category {cat_idx}/{category_count}: {category} ({len(category_questions)} questions)")
        
        for q_idx, question in enumerate(category_questions, 1):
            try:
                print(f"   🔎 Analyzing question {q_idx}/{len(category_questions)}: {question[:50]}...")
                embeddings = analyzer.get_thought_embeddings(question)
                analyzer.visualize_thought_space(embeddings, category, output_dir=results_dir)
                all_embeddings.append(embeddings)
                categories.append(category)
                total_processed += 1
            except Exception as e:
                print(f"   ❗ Error processing question: {question[:50]}... ({e})")
    
    print("\n📊 Creating visualizations...")
    if all_embeddings:
        print(f"🧩 Combining {len(all_embeddings)} question embeddings")
        combined_embeddings = np.concatenate(all_embeddings)
        analyzer.visualize_thought_space(combined_embeddings, "All_Categories", output_dir=results_dir)
        print(f"📈 Saved combined visualization: {results_dir}/thought_trajectory_All_Categories.png")
    else:
        print("⚠️ No valid embeddings generated - nothing to visualize")
    
    print(f"\n✅ Analysis complete! Processed {total_processed} questions across {category_count} categories")

if __name__ == "__main__":
    main()

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", device_map="auto")

# Create the pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,  
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