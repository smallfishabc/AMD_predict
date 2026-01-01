# SimpleFold Inference-Only Modification Plan

## Goal
Modify SimpleFold to accept pre-computed ESM2 embeddings and perform batch-wise inference with minimal code changes.

## Architecture Overview

```
Input: Pre-computed ESM2 embeddings (.pt files)
  ↓
SimpleFold Model (structure prediction)
  ↓
Output: PDB structures (batch)
```

## File Structure

```
simplefold/
├── inference/
│   ├── batch_inference.py      # NEW: Batch inference wrapper
│   └── embedding_loader.py     # NEW: Load pre-computed embeddings
└── model/
    └── simplefold.py           # MODIFY: Accept embeddings directly
```

## Required Changes

### 1. Create Embedding Loader (NEW FILE)
**File:** `simplefold/inference/embedding_loader.py`

```python
# Purpose: Load pre-computed ESM2 embeddings
# Input: Path to .pt file or directory
# Output: Tensor [batch_size, seq_len, 2560]

class EmbeddingLoader:
    def load_single(path):
        # Load single protein embedding
        pass
    
    def load_batch(paths):
        # Load multiple embeddings, pad to max length
        pass
    
    def collate(embeddings):
        # Pad sequences to same length for batching
        pass
```

### 2. Create Batch Inference Script (NEW FILE)
**File:** `simplefold/inference/batch_inference.py`

```python
# Purpose: Process multiple proteins in batch
# Input: List of embedding paths
# Output: List of PDB structures

def batch_predict(
    embedding_paths,      # List of .pt files
    model,                # SimpleFold model
    batch_size=8,         # Adaptive based on length
    output_dir='./outputs'
):
    # 1. Load embeddings in batches
    # 2. Pad to max length in batch
    # 3. Run SimpleFold inference
    # 4. Save PDB files
    pass
```

### 3. Modify SimpleFold Model (MINIMAL CHANGE)
**File:** `simplefold/model/simplefold.py`

Find the `forward()` or `predict()` method that currently does:
```python
def predict(self, sequence):
    # OLD: Compute ESM2 embeddings
    embeddings = self.esm2_model(sequence)
    
    # Run structure prediction
    structure = self.model(embeddings)
    return structure
```

Change to:
```python
def predict(self, sequence=None, embeddings=None):
    # NEW: Accept pre-computed embeddings OR sequence
    if embeddings is None:
        embeddings = self.esm2_model(sequence)
    
    # Run structure prediction (unchanged)
    structure = self.model(embeddings)
    return structure
```

**That's it!** Just add one optional parameter.

### 4. Create CLI Script (NEW FILE)
**File:** `run_batch_inference.py` (in AMD_predict repo)

```python
#!/usr/bin/env python3
"""
Run SimpleFold inference with pre-computed ESM2 embeddings
Usage:
    python run_batch_inference.py \
        --embeddings_dir ./embeddings_output/bin8 \
        --output_dir ./structures_output/bin8 \
        --batch_size 4 \
        --model simplefold_1.6B
"""

import argparse
from simplefold.inference import batch_inference
from simplefold.model import load_model

def main():
    args = parse_args()
    
    # Load SimpleFold model
    model = load_model(args.model)
    
    # Get embedding files
    embedding_files = glob(f"{args.embeddings_dir}/*.pt")
    
    # Run batch inference
    batch_inference.batch_predict(
        embedding_paths=embedding_files,
        model=model,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )
```

## Implementation Steps

### Step 1: Find SimpleFold Entry Points
```bash
# In your SimpleFold repo, find:
grep -r "def predict" --include="*.py"
grep -r "def forward" --include="*.py" | grep -i "fold"
grep -r "esm" --include="*.py" | grep -i "embed"
```

**Look for:**
- Main inference function
- Where ESM2 is called
- Input/output handling

### Step 2: Create Embedding Loader
```python
# simplefold/inference/embedding_loader.py
import torch
import glob
from pathlib import Path

class EmbeddingLoader:
    @staticmethod
    def load_single(path):
        """Load single embedding file"""
        return torch.load(path)
    
    @staticmethod
    def load_batch(paths, max_length=None):
        """Load multiple embeddings and pad"""
        embeddings = [torch.load(p) for p in paths]
        
        # Find max length
        if max_length is None:
            max_length = max(e.shape[1] for e in embeddings)
        
        # Pad all to max_length
        padded = []
        masks = []
        for emb in embeddings:
            seq_len = emb.shape[1]
            if seq_len < max_length:
                pad = torch.zeros(1, max_length - seq_len, emb.shape[2])
                emb = torch.cat([emb, pad], dim=1)
            
            mask = torch.ones(max_length, dtype=torch.bool)
            mask[seq_len:] = False
            
            padded.append(emb)
            masks.append(mask)
        
        return torch.cat(padded, dim=0), torch.stack(masks)
```

### Step 3: Modify SimpleFold Predict Function
**Location:** Find where SimpleFold computes embeddings

**Change:**
```python
# BEFORE
def predict_structure(self, sequence: str):
    tokens = self.tokenize(sequence)
    embeddings = self.esm_model(tokens)
    structure = self.fold_model(embeddings)
    return structure

# AFTER (add one parameter)
def predict_structure(self, sequence: str = None, embeddings: torch.Tensor = None):
    if embeddings is None:
        tokens = self.tokenize(sequence)
        embeddings = self.esm_model(tokens)
    structure = self.fold_model(embeddings)
    return structure
```

### Step 4: Create Batch Inference Wrapper
```python
# simplefold/inference/batch_inference.py
import torch
from pathlib import Path
from .embedding_loader import EmbeddingLoader

def batch_predict(model, embedding_paths, batch_size=8, output_dir='./outputs'):
    """
    Run batch inference on pre-computed embeddings
    
    Args:
        model: SimpleFold model instance
        embedding_paths: List of .pt file paths
        batch_size: Number of proteins per batch
        output_dir: Where to save PDB files
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    loader = EmbeddingLoader()
    
    # Process in batches
    for i in range(0, len(embedding_paths), batch_size):
        batch_paths = embedding_paths[i:i+batch_size]
        
        # Load embeddings
        embeddings, masks = loader.load_batch(batch_paths)
        
        # Run inference
        with torch.no_grad():
            structures = model.predict_structure(embeddings=embeddings)
        
        # Save each structure
        for j, (path, structure) in enumerate(zip(batch_paths, structures)):
            protein_id = Path(path).stem
            output_path = Path(output_dir) / f"{protein_id}.pdb"
            save_pdb(structure, output_path, mask=masks[j])
            
        print(f"Processed batch {i//batch_size + 1}/{len(embedding_paths)//batch_size + 1}")
```

### Step 5: Adaptive Batch Sizing
```python
# Add to batch_inference.py
def get_adaptive_batch_size(avg_seq_length):
    """Match our length-based batching strategy"""
    if avg_seq_length < 100:
        return 256
    elif avg_seq_length < 200:
        return 256
    elif avg_seq_length < 300:
        return 256
    elif avg_seq_length < 400:
        return 166
    elif avg_seq_length < 500:
        return 108
    elif avg_seq_length < 1000:
        return 44
    elif avg_seq_length < 2000:
        return 12
    else:
        return 2
```

## Integration with AMD_predict Pipeline

### Workflow
```bash
# 1. Generate ESM2 embeddings (1 hour)
python generate_esm2_embeddings.py \
    --input sequences_by_length/bin8_2000plus_aa/bin8_2000plus_aa.fasta \
    --output embeddings_output/bin8/ \
    --batch_size 2

# 2. Run SimpleFold with pre-computed embeddings (48 hours)
python run_batch_inference.py \
    --embeddings_dir embeddings_output/bin8/ \
    --output_dir structures_output/bin8/ \
    --batch_size 2 \
    --model simplefold_1.6B
```

## Testing Plan

### Test 1: Single Protein
```python
# Test loading one embedding and predicting
embedding = torch.load('embeddings_output/bin8/protein_001.pt')
structure = model.predict_structure(embeddings=embedding)
assert structure is not None
```

### Test 2: Small Batch
```python
# Test batch of 4 proteins
paths = ['embeddings_output/bin1/protein_{:03d}.pt'.format(i) for i in range(4)]
batch_predict(model, paths, batch_size=4, output_dir='test_outputs')
assert len(os.listdir('test_outputs')) == 4
```

### Test 3: Different Lengths
```python
# Test padding works correctly
paths = [
    'embeddings_output/bin1/short_100aa.pt',
    'embeddings_output/bin3/medium_300aa.pt',
    'embeddings_output/bin8/long_2500aa.pt'
]
# Should pad all to 2500 and handle correctly
```

## Files to Create in AMD_predict Repo

1. **generate_esm2_embeddings.py** - Generate and save embeddings
2. **run_batch_inference.py** - Run SimpleFold on embeddings
3. **embedding_loader.py** - Utility to load/batch embeddings

## Expected Performance

Based on SimpleFold-1.6B benchmarks:
- Bin8 (2000+ aa): ~2 proteins/hour with batch_size=2
- Bin7 (1000-1999 aa): ~6 proteins/hour with batch_size=12
- Bin6 (500-999 aa): ~30 proteins/hour with batch_size=44

## Summary of Changes

**SimpleFold repo (minimal):**
- ✅ 1 parameter added to predict function: `embeddings=None`
- ✅ 2 new utility files (embedding_loader.py, batch_inference.py)

**AMD_predict repo:**
- ✅ ESM2 embedding generation script
- ✅ Batch inference runner
- ✅ Integration with existing bin structure

**Total new code:** ~300 lines
**Modified existing code:** ~5 lines

This is the minimal viable approach!
