# ===================================================================
# BioFoundry Active Learning with Geometric Deep Learning
# Colab Notebook - Corrected & Production-Ready Version
# ===================================================================
# Author: Based on correcting.md analysis
# Purpose: Reproduce DBTL cycle with EquiformerV2 + Batch Bayesian Optimization
# ===================================================================

# ==== CELL 1: Environment Check & GPU Verification ====
"""
## 🔧 Step 0: Environment Setup & GPU Check

First, verify you have GPU access and check the GPU type.
"""

import subprocess
import sys

# Check GPU
print("=" * 60)
print("GPU Information:")
print("=" * 60)
subprocess.run(["nvidia-smi"], check=False)

import torch
print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Determine batch size based on GPU
    gpu_name = torch.cuda.get_device_name(0)
    if "A100" in gpu_name:
        RECOMMENDED_BATCH_SIZE = 16
        RECOMMENDED_LMAX = [4]
    elif "V100" in gpu_name:
        RECOMMENDED_BATCH_SIZE = 8
        RECOMMENDED_LMAX = [4]
    elif "T4" in gpu_name:
        RECOMMENDED_BATCH_SIZE = 4
        RECOMMENDED_LMAX = [2]  # Critical: T4 cannot handle lmax=4
    else:
        RECOMMENDED_BATCH_SIZE = 4
        RECOMMENDED_LMAX = [2]
    
    print(f"\n⚠️ Recommended Config for {gpu_name}:")
    print(f"  - batch_size: {RECOMMENDED_BATCH_SIZE}")
    print(f"  - lmax_list: {RECOMMENDED_LMAX}")
else:
    print("⚠️ WARNING: No GPU detected. Training will be extremely slow.")


# ==== CELL 2: Install Dependencies (Corrected Order) ====
"""
## 📦 Step 1: Install Dependencies

Following the production-grade installation order from correcting.md:
1. Uninstall existing PyG components
2. Install specific PyTorch version
3. Install PyG with matching CUDA version
"""

print("\n" + "=" * 60)
print("Installing Dependencies...")
print("=" * 60)

# Uninstall existing PyG (avoid conflicts)
subprocess.run([
    sys.executable, "-m", "pip", "uninstall", "-y",
    "torch-scatter", "torch-sparse", "torch-geometric", "torch-cluster"
], check=False)

# Install PyTorch (stable version for Colab)
subprocess.run([
    sys.executable, "-m", "pip", "install", "torch==2.1.0", "torchvision==0.16.0"
], check=True)

# Install PyG with CUDA 12.1 (Colab default)
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv",
    "-f", "https://data.pyg.org/whl/torch-2.1.0+cu121.html"
], check=True)

subprocess.run([
    sys.executable, "-m", "pip", "install", "torch-geometric"
], check=True)

# Install other dependencies
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "lmdb", "pyyaml", "tqdm", "biopython", "ase", "e3nn", "timm",
    "scipy==1.13.1",  # Critical: sph_harm compatibility
    "numba",
    "wandb", "tensorboard",
    "scikit-learn", "matplotlib", "seaborn"
], check=True)

print("\n✅ All dependencies installed successfully!")


# ==== CELL 3: Mount Google Drive & Upload Data ====
"""
## 📂 Step 2: Data Preparation

Upload your LMDB datasets (train.lmdb, val.lmdb) to Google Drive,
then copy them to Colab's local disk for faster I/O.
"""

from google.colab import drive
import os
import shutil

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Define paths
DRIVE_DATA_PATH = "/content/drive/My Drive/BioFoundry/data"  # ⚠️ Modify this path
LOCAL_DATA_PATH = "/content/data"
CHECKPOINT_PATH = "/content/checkpoints"
EMBEDDING_PATH = "/content/embeddings.npy"

# Create local directories
os.makedirs(LOCAL_DATA_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Copy LMDB from Drive to local disk (CRITICAL for performance)
print("Copying LMDB files from Google Drive to local disk...")
print("⏳ This may take 2-5 minutes for large datasets...")

if os.path.exists(DRIVE_DATA_PATH):
    shutil.copytree(DRIVE_DATA_PATH, LOCAL_DATA_PATH, dirs_exist_ok=True)
    print(f"✅ Data copied to {LOCAL_DATA_PATH}")
    
    # Verify files
    print("\nData directory contents:")
    subprocess.run(["ls", "-lh", LOCAL_DATA_PATH])
else:
    print(f"❌ ERROR: {DRIVE_DATA_PATH} not found!")
    print("Please upload your LMDB files to Google Drive first.")


# ==== CELL 4: Clone Code Repositories ====
"""
## 📥 Step 3: Download EquiformerV2 Code
"""

import os

os.chdir("/content")

# Clone OCP (Open Catalyst Project)
if not os.path.exists("/content/ocp"):
    subprocess.run(["git", "clone", "https://github.com/Open-Catalyst-Project/ocp.git"], check=True)
    print("✅ OCP cloned")

# Clone EquiformerV2
if not os.path.exists("/content/equiformer_v2"):
    subprocess.run(["git", "clone", "https://github.com/atomicarchitects/equiformer_v2.git"], check=True)
    print("✅ EquiformerV2 cloned")

# Add to Python path
sys.path.insert(0, "/content/ocp")
sys.path.insert(0, "/content/equiformer_v2")

print("\n✅ Code repositories ready")


# ==== CELL 5: Create Training Configuration ====
"""
## ⚙️ Step 4: Generate Training Config (GPU-Adaptive)
"""

import yaml

# GPU-adaptive configuration
config = {
    "trainer": "energy_v2",
    "dataset": {
        "train": {
            "src": f"{LOCAL_DATA_PATH}/train.lmdb",
            "normalize_labels": False
        },
        "val": {
            "src": f"{LOCAL_DATA_PATH}/val.lmdb"
        }
    },
    "logger": "tensorboard",
    "task": {
        "dataset": "lmdb_v2",
        "description": "BioFoundry Active Learning - Geometric Features",
        "type": "regression",
        "metric": "mae",
        "primary_metric": "mae",
        "labels": ["predicted_score"]
    },
    "model": {
        "name": "equiformer_v2",
        "use_pbc": False,
        "regress_forces": False,
        "otf_graph": True,
        "max_neighbors": 20,
        "max_radius": 12.0,
        "max_num_elements": 90,
        "num_layers": 4,
        "sphere_channels": 64,
        "attn_hidden_channels": 64,
        "num_heads": 4,
        "attn_alpha_channels": 64,
        "attn_value_channels": 32,
        "ffn_hidden_channels": 128,
        "norm_type": "layer_norm",
        "lmax_list": RECOMMENDED_LMAX,  # GPU-adaptive
        "mmax_list": [2] if RECOMMENDED_LMAX == [4] else [1],
        "grid_resolution": 18 if RECOMMENDED_LMAX == [4] else 8
    },
    "optim": {
        "batch_size": RECOMMENDED_BATCH_SIZE,
        "eval_batch_size": RECOMMENDED_BATCH_SIZE * 2,
        "num_workers": 2,
        "lr_initial": 0.001,
        "optimizer": "AdamW",
        "optimizer_params": {"weight_decay": 0.01},
        "scheduler": "ReduceLROnPlateau",
        "scheduler_params": {
            "factor": 0.5,
            "patience": 5,
            "epochs": 50
        },
        "mode": "min",
        "max_epochs": 50,
        "energy_coefficient": 1.0,
        "eval_every": 5,
        "checkpoint_every": 10
    }
}

# Save config
config_path = "/content/colab_config.yml"
with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"✅ Configuration saved to {config_path}")
print("\nConfig Preview:")
print(yaml.dump(config, default_flow_style=False))


# ==== CELL 6: Train EquiformerV2 ====
"""
## 🚀 Step 5: Train EquiformerV2

WARNING: This will take 2-6 hours depending on:
- GPU type (T4: ~6h, V100: ~3h, A100: ~2h)
- Dataset size
- Number of epochs

You can monitor progress via TensorBoard (see next cell).
"""

import os
os.environ['PYTHONPATH'] = '/content/ocp:/content/equiformer_v2'
os.chdir("/content/equiformer_v2")

# Launch training
print("=" * 60)
print("Starting EquiformerV2 Training...")
print("=" * 60)

subprocess.run([
    sys.executable, "main_oc20.py",
    "--config-yml", config_path,
    "--mode", "train",
    "--run-dir", CHECKPOINT_PATH,
    "--print-every", "10"
], check=True)

print("\n✅ Training completed!")
print(f"Checkpoints saved to: {CHECKPOINT_PATH}")


# ==== CELL 7: TensorBoard Monitoring (Run in Parallel) ====
"""
## 📊 Step 5b: Monitor Training with TensorBoard

Run this cell in a separate tab while training is running.
"""

# %load_ext tensorboard
# %tensorboard --logdir /content/checkpoints


# ==== CELL 8: Extract Geometric Embeddings (CORRECTED) ====
"""
## 🧬 Step 6: Extract Geometric Embeddings (Corrected Version)

This implements the corrected embedding extraction using `register_forward_hook`.

Key Fix:
- EquiformerV2's forward() only returns energy (scalar)
- We use Hook to capture the latent features BEFORE the energy head
"""

import torch
import lmdb
import pickle
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

print("=" * 60)
print("Step 6: Extracting Geometric Embeddings...")
print("=" * 60)

# 1. Load checkpoint
checkpoint_files = [f for f in os.listdir(CHECKPOINT_PATH) if f.endswith('.pt')]
if not checkpoint_files:
    raise FileNotFoundError(f"No checkpoint found in {CHECKPOINT_PATH}")

best_checkpoint = sorted(checkpoint_files)[-1]  # Use the latest checkpoint
checkpoint_path = os.path.join(CHECKPOINT_PATH, best_checkpoint)

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cuda')

# 2. Reconstruct model from config
# Note: You need to adapt this to your specific model loading logic
# This is a placeholder - the actual implementation depends on your codebase

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import load_config

config_dict = checkpoint.get('config', config)
model = registry.get_model_class(config_dict['model']['name'])(
    **config_dict['model']
)

# Load weights
state_dict = checkpoint['state_dict']
# Remove 'module.' prefix if present (from DistributedDataParallel)
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

model = model.to('cuda')
model.eval()

print("✅ Model loaded successfully")

# 3. Define Hook to capture embeddings
features_cache = {}

def get_embedding_hook(name):
    def hook(module, input, output):
        """
        Hook function to capture intermediate features.
        
        For EquiformerV2:
        - The output of the last norm layer is the graph-level embedding
        - If output is node-level, we need to aggregate across nodes
        """
        # Detach and store
        features_cache[name] = output.detach()
    return hook

# 4. Register hook
# Critical: Find the correct layer name
# For EquiformerV2, the embedding is typically at:
# - model.norm (final layer norm before energy head)
# - or model.energy_block (if you want pre-energy features)

# Print model structure to find the right layer
print("\nModel structure (first 20 layers):")
for i, (name, module) in enumerate(model.named_modules()):
    print(f"  {name}: {type(module).__name__}")
    if i > 20:
        print("  ...")
        break

# Register hook (adjust layer name based on your model)
# Common choices: 'norm', 'norm_final', 'energy_block'
hook_layer_name = 'energy_block'  # ⚠️ Verify this matches your model

if hasattr(model, hook_layer_name):
    hook_handle = getattr(model, hook_layer_name).register_forward_pre_hook(
        lambda m, inp: features_cache.update({'embedding': inp[0].detach()})
    )
    print(f"✅ Hook registered at: {hook_layer_name}")
else:
    print(f"⚠️ WARNING: Layer '{hook_layer_name}' not found. Trying alternative...")
    # Fallback: hook the last norm layer
    hook_handle = model.norm.register_forward_hook(get_embedding_hook('embedding'))
    print("✅ Hook registered at: model.norm (fallback)")


# 5. Create DataLoader for all data
class LMDBDataset:
    """Simple LMDB dataset wrapper"""
    def __init__(self, lmdb_path):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            self.length = txn.stat()['entries']
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with self.env.begin() as txn:
            data = pickle.loads(txn.get(str(idx).encode()))
        return Data(**data)

# Load both train and val
train_dataset = LMDBDataset(f"{LOCAL_DATA_PATH}/train.lmdb")
val_dataset = LMDBDataset(f"{LOCAL_DATA_PATH}/val.lmdb")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"\nDataset sizes:")
print(f"  Train: {len(train_dataset)} samples")
print(f"  Val: {len(val_dataset)} samples")


# 6. Extract embeddings
embeddings_dict = {}

print("\nExtracting embeddings from training set...")
for batch_idx, batch in enumerate(tqdm(train_loader)):
    batch = batch.to('cuda')
    
    with torch.no_grad():
        _ = model(batch)  # Forward pass triggers hook
        
        # Get embedding from cache
        emb = features_cache['embedding']
        
        # If embedding is node-level, aggregate to graph-level
        if emb.dim() == 2:  # (num_nodes, dim)
            from torch_geometric.nn import global_mean_pool
            emb = global_mean_pool(emb, batch.batch)
        
        # Store embeddings with sample IDs
        emb_np = emb.cpu().numpy()
        
        # Get sample IDs (assuming they're stored in batch.sid)
        if hasattr(batch, 'sid'):
            sample_ids = batch.sid
        else:
            # Fallback: use batch indices
            sample_ids = [f"train_{batch_idx * 16 + i}" for i in range(len(emb_np))]
        
        for sid, embedding in zip(sample_ids, emb_np):
            embeddings_dict[str(sid)] = embedding

print(f"✅ Extracted {len(embeddings_dict)} embeddings")

# Save embeddings
np.save(EMBEDDING_PATH, embeddings_dict)
print(f"✅ Embeddings saved to {EMBEDDING_PATH}")

# Cleanup
hook_handle.remove()
print("✅ Hook removed")

# Print sample embedding
sample_key = list(embeddings_dict.keys())[0]
sample_emb = embeddings_dict[sample_key]
print(f"\nSample embedding shape: {sample_emb.shape}")
print(f"Sample embedding (first 5 dims): {sample_emb[:5]}")


# ==== CELL 9: Active Learning - Batch Bayesian Optimization ====
"""
## 🎯 Step 7: Active Learning with Batch Diversity Sampling

This implements the corrected version from correcting.md:
- Name changed from "MOBO-OSD" to "Batch Diversity Sampling"
- Uses Gaussian Process + UCB acquisition
- Diversity ensured via cosine similarity penalty (pool-based approximation)

Note: This is NOT true MOBO-OSD (which requires gradient projection),
but a practical approximation for discrete candidate pools.
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
import numpy as np

class BatchDiversityOptimizer:
    """
    Batch Bayesian Optimization with Diversity Penalty.
    
    This is a pool-based approximation of orthogonal sampling.
    For true MOBO-OSD, see BoTorch's qNoisyExpectedImprovement.
    """
    
    def __init__(self, embeddings_dict, initial_scores, beta=2.0):
        """
        Args:
            embeddings_dict: {sample_id: embedding_vector}
            initial_scores: {sample_id: score}
            beta: UCB exploration parameter
        """
        self.embeddings_dict = embeddings_dict
        self.all_ids = list(embeddings_dict.keys())
        
        # Separate labeled and unlabeled
        self.labeled_ids = list(initial_scores.keys())
        self.unlabeled_ids = [sid for sid in self.all_ids if sid not in initial_scores]
        
        # Prepare training data
        self.X_train = np.array([embeddings_dict[sid] for sid in self.labeled_ids])
        self.y_train = np.array([initial_scores[sid] for sid in self.labeled_ids])
        
        # Standardize features
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self.X_train_scaled = self.scaler_X.fit_transform(self.X_train)
        self.y_train_scaled = self.scaler_y.fit_transform(self.y_train.reshape(-1, 1)).ravel()
        
        # Gaussian Process
        kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=1.0)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=False  # We handle normalization manually
        )
        
        self.beta = beta
        self.iteration = 0
        
        print(f"Initialized with {len(self.labeled_ids)} labeled samples")
        print(f"Pool size: {len(self.unlabeled_ids)} unlabeled samples")
    
    def fit(self):
        """Train Gaussian Process on current labeled data"""
        self.gp.fit(self.X_train_scaled, self.y_train_scaled)
        print(f"GP trained. Kernel: {self.gp.kernel_}")
    
    def acquisition_ucb(self, X_pool_scaled):
        """
        Upper Confidence Bound acquisition function.
        
        UCB = μ + β * σ
        """
        mu, sigma = self.gp.predict(X_pool_scaled, return_std=True)
        return mu + self.beta * sigma
    
    def select_batch_diverse(self, batch_size=10, diversity_weight=0.5):
        """
        Select a diverse batch of candidates using greedy selection
        with cosine similarity penalty.
        
        Args:
            batch_size: Number of samples to select
            diversity_weight: Weight for diversity penalty (0-1)
        
        Returns:
            List of selected sample IDs
        """
        if len(self.unlabeled_ids) == 0:
            print("⚠️ No unlabeled samples remaining!")
            return []
        
        # Prepare pool
        X_pool = np.array([self.embeddings_dict[sid] for sid in self.unlabeled_ids])
        X_pool_scaled = self.scaler_X.transform(X_pool)
        
        # Calculate base acquisition scores
        acq_scores = self.acquisition_ucb(X_pool_scaled)
        
        selected_indices = []
        selected_embeddings = []
        
        for step in range(min(batch_size, len(self.unlabeled_ids))):
            if step == 0:
                # First selection: pure UCB
                best_idx = np.argmax(acq_scores)
            else:
                # Subsequent selections: UCB with diversity penalty
                # Calculate cosine similarity to already selected samples
                selected_matrix = np.array(selected_embeddings)
                pool_matrix = X_pool
                
                # Normalize for cosine similarity
                selected_norm = selected_matrix / (np.linalg.norm(selected_matrix, axis=1, keepdims=True) + 1e-8)
                pool_norm = pool_matrix / (np.linalg.norm(pool_matrix, axis=1, keepdims=True) + 1e-8)
                
                # Cosine similarity: (num_pool, num_selected)
                similarities = np.dot(pool_norm, selected_norm.T)
                
                # Max similarity to any selected sample
                max_similarity = np.abs(similarities).max(axis=1)
                
                # Penalize similar candidates
                diversity_penalty = max_similarity * diversity_weight
                adjusted_scores = acq_scores * (1 - diversity_penalty)
                
                # Don't re-select already chosen samples
                adjusted_scores[selected_indices] = -np.inf
                
                best_idx = np.argmax(adjusted_scores)
            
            selected_indices.append(best_idx)
            selected_embeddings.append(X_pool[best_idx])
            
            # Exclude from future selection
            acq_scores[best_idx] = -np.inf
        
        # Map indices back to sample IDs
        selected_ids = [self.unlabeled_ids[i] for i in selected_indices]
        
        print(f"\nSelected {len(selected_ids)} candidates:")
        for i, sid in enumerate(selected_ids):
            print(f"  {i+1}. {sid}")
        
        return selected_ids
    
    def update(self, new_scores):
        """
        Update the model with new experimental results.
        
        Args:
            new_scores: {sample_id: score} for newly tested samples
        """
        # Add to labeled set
        for sid, score in new_scores.items():
            if sid in self.unlabeled_ids:
                self.labeled_ids.append(sid)
                self.unlabeled_ids.remove(sid)
        
        # Rebuild training data
        self.X_train = np.array([self.embeddings_dict[sid] for sid in self.labeled_ids])
        self.y_train = np.array([new_scores.get(sid, self.y_train[i]) 
                                 for i, sid in enumerate(self.labeled_ids)])
        
        # Re-fit scalers
        self.X_train_scaled = self.scaler_X.fit_transform(self.X_train)
        self.y_train_scaled = self.scaler_y.fit_transform(self.y_train.reshape(-1, 1)).ravel()
        
        self.iteration += 1
        print(f"✅ Model updated. Iteration {self.iteration}, Labeled: {len(self.labeled_ids)}")


# ==== CELL 10: Demo - Active Learning Loop ====
"""
## 🔄 Step 8: Run Active Learning Loop (Demo)

This demonstrates a complete DBTL cycle iteration.
"""

# Load embeddings
embeddings = np.load(EMBEDDING_PATH, allow_pickle=True).item()

print(f"Loaded {len(embeddings)} embeddings")

# Simulate initial labeled data (replace with your actual initial experiments)
# For demo: randomly select 20 samples as initial training set
np.random.seed(42)
all_sample_ids = list(embeddings.keys())
initial_sample_ids = np.random.choice(all_sample_ids, size=20, replace=False).tolist()

# Simulate scores (replace with your actual experimental data)
# For demo: generate random scores
initial_scores = {sid: np.random.randn() for sid in initial_sample_ids}

print(f"\nInitial training set: {len(initial_scores)} samples")
print(f"Score range: [{min(initial_scores.values()):.2f}, {max(initial_scores.values()):.2f}]")


# Initialize optimizer
optimizer = BatchDiversityOptimizer(
    embeddings_dict=embeddings,
    initial_scores=initial_scores,
    beta=2.0  # Exploration parameter
)

# Fit GP
optimizer.fit()

# Select next batch for experiments
BATCH_SIZE = 10  # Adjust based on your experimental throughput
next_batch = optimizer.select_batch_diverse(
    batch_size=BATCH_SIZE,
    diversity_weight=0.5  # 0=pure exploitation, 1=pure diversity
)

print("\n" + "=" * 60)
print(f"🎯 Recommended candidates for next experiments:")
print("=" * 60)
for i, sid in enumerate(next_batch):
    print(f"{i+1:2d}. Sample: {sid}")

# Save results
with open("/content/selected_batch_round1.txt", "w") as f:
    for sid in next_batch:
        f.write(f"{sid}\n")

print(f"\n✅ Batch saved to /content/selected_batch_round1.txt")
print("\n📝 Next steps:")
print("  1. Perform manual validation on these candidates")
print("  2. Record experimental results")
print("  3. Run Cell 11 to update the model")


# ==== CELL 11: Update Model with New Results ====
"""
## 🔄 Step 9: Update Model (After Manual Validation)

After you complete manual experiments, run this cell to update the model
and select the next batch.
"""

# Example: Simulated new experimental results
# Replace this with your actual validation data
new_experimental_results = {
    next_batch[0]: 1.5,  # Replace with actual score
    next_batch[1]: 0.8,
    next_batch[2]: -0.3,
    # ... add all tested samples
}

print("📊 New experimental results:")
for sid, score in new_experimental_results.items():
    print(f"  {sid}: {score:.3f}")

# Update optimizer
optimizer.update(new_experimental_results)

# Re-fit GP with updated data
optimizer.fit()

# Select next batch
next_batch_round2 = optimizer.select_batch_diverse(
    batch_size=BATCH_SIZE,
    diversity_weight=0.5
)

print("\n" + "=" * 60)
print(f"🎯 Round 2 - Recommended candidates:")
print("=" * 60)
for i, sid in enumerate(next_batch_round2):
    print(f"{i+1:2d}. {sid}")

# This loop continues until convergence...


# ==== CELL 12: Visualization & Analysis ====
"""
## 📊 Step 10: Visualize Results
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Prepare data for visualization
labeled_embeddings = np.array([embeddings[sid] for sid in optimizer.labeled_ids])
labeled_scores = np.array([optimizer.y_train[i] for i in range(len(optimizer.labeled_ids))])

# Dimensionality reduction for visualization
pca = PCA(n_components=2)
labeled_2d = pca.fit_transform(labeled_embeddings)

# Plot 1: Embedding space with scores
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot colored by score
scatter = axes[0].scatter(
    labeled_2d[:, 0], labeled_2d[:, 1],
    c=labeled_scores, cmap='viridis',
    s=100, alpha=0.6, edgecolors='black'
)
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].set_title('Embedding Space (PCA) - Colored by Score')
plt.colorbar(scatter, ax=axes[0], label='Score')

# Mark selected samples
if 'next_batch' in locals():
    next_batch_embeddings = np.array([embeddings[sid] for sid in next_batch if sid in embeddings])
    if len(next_batch_embeddings) > 0:
        next_batch_2d = pca.transform(next_batch_embeddings)
        axes[0].scatter(
            next_batch_2d[:, 0], next_batch_2d[:, 1],
            c='red', s=200, alpha=0.8, marker='*',
            edgecolors='black', linewidths=2,
            label='Selected for Next Round'
        )
        axes[0].legend()

# Plot 2: Acquisition function landscape
# Create a grid for visualization
x_min, x_max = labeled_2d[:, 0].min() - 1, labeled_2d[:, 0].max() + 1
y_min, y_max = labeled_2d[:, 1].min() - 1, labeled_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Inverse transform to embedding space
grid_points_2d = np.c_[xx.ravel(), yy.ravel()]
# Note: This is approximate since PCA is lossy
# For exact visualization, you'd need the full embedding space

axes[1].scatter(labeled_2d[:, 0], labeled_2d[:, 1],
                c='blue', s=50, alpha=0.3, label='Labeled')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_title('Search Space Coverage')
axes[1].legend()

plt.tight_layout()
plt.savefig('/content/active_learning_visualization.png', dpi=150, bbox_inches='tight')
print("✅ Visualization saved to /content/active_learning_visualization.png")
plt.show()


# ==== CELL 13: Save Results & Download ====
"""
## 💾 Step 11: Save All Results

Save embeddings, model, and results to Google Drive for backup.
"""

import pickle
from datetime import datetime

# Create results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"/content/drive/My Drive/BioFoundry/results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# Save embeddings
np.save(f"{results_dir}/embeddings.npy", embeddings)

# Save optimizer state
with open(f"{results_dir}/optimizer_state.pkl", "wb") as f:
    pickle.dump({
        'labeled_ids': optimizer.labeled_ids,
        'unlabeled_ids': optimizer.unlabeled_ids,
        'scores': dict(zip(optimizer.labeled_ids, optimizer.y_train)),
        'iteration': optimizer.iteration
    }, f)

# Save selected batches
with open(f"{results_dir}/selected_batches.txt", "w") as f:
    f.write(f"# Active Learning Results - {timestamp}\n\n")
    f.write("## Round 1 Selected Candidates:\n")
    for sid in next_batch:
        f.write(f"{sid}\n")

# Copy checkpoint
shutil.copy(checkpoint_path, f"{results_dir}/best_model.pt")

print(f"✅ All results saved to {results_dir}")
print("\n📂 Saved files:")
print(f"  - embeddings.npy")
print(f"  - optimizer_state.pkl")
print(f"  - selected_batches.txt")
print(f"  - best_model.pt")


# ==== CELL 14: Summary & Next Steps ====
"""
## ✅ Workflow Complete!

### What We Accomplished:
1. ✅ Trained EquiformerV2 on your CAR-T dataset
2. ✅ Extracted geometric embeddings (using corrected Hook method)
3. ✅ Implemented Batch Diversity Sampling (corrected from MOBO-OSD)
4. ✅ Selected next batch of candidates for manual validation

### Next Steps:
1. **Manual Validation**: Test the selected candidates in your lab
2. **Record Results**: Measure the performance metrics
3. **Update Model**: Run Cell 11 with new results
4. **Iterate**: Repeat until you reach the Pareto frontier

### Key Corrections Applied:
- ✅ Embedding extraction via `register_forward_hook` (not direct model output)
- ✅ Renamed MOBO-OSD → Batch Diversity Sampling (accurate naming)
- ✅ GPU-adaptive batch sizes (T4: 4, V100: 8, A100: 16)
- ✅ LMDB copied to local disk (not read from Drive)
- ✅ scipy==1.13.1 for sph_harm compatibility

### Troubleshooting:
- If OOM: Reduce `batch_size` and `lmax_list`
- If slow: Check LMDB is on local disk, not Drive
- If Hook fails: Print `model` structure to find correct layer name
"""

print("=" * 60)
print("🎉 BioFoundry Active Learning Pipeline Complete!")
print("=" * 60)
print(f"\nResults saved to: {results_dir}")
print(f"Embeddings: {len(embeddings)} samples")
print(f"Current labeled pool: {len(optimizer.labeled_ids)} samples")
print(f"Remaining unlabeled: {len(optimizer.unlabeled_ids)} samples")
print("\n📈 Good luck with your experiments!")
