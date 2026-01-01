# Testing Docker on DigitalOcean AMD Cloud

## Quick Cloud Test (5-10 minutes)

### Step 1: Launch MI300X Instance
```bash
# On DigitalOcean AMD cloud:
# 1. Create instance with MI300X GPU
# 2. SSH into instance
ssh root@your-instance-ip
```

### Step 2: Clone and Test
```bash
# Clone your repo
git clone https://github.com/smallfishabc/AMD_predict.git
cd AMD_predict

# Quick environment test
./test_environment.sh
# Expected: ✅ ROCm found, ✅ GPU devices, ✅ Docker found
```

### Step 3: Build Docker Image
```bash
# Build lightweight version first (faster, ~10 min)
docker build -t amd-protein:test -f Dockerfile.lightweight .

# Check build success
docker images | grep amd-protein
```

### Step 4: Run Quick Test
```bash
# Start container
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  --shm-size=16g \
  -v $(pwd):/workspace \
  amd-protein:test

# Inside container, test GPU
rocm-smi
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB')"

# Test imports
python -c "import torch, transformers, Bio; print('✓ All packages working')"

# Exit container
exit
```

### Step 5: Test SimpleFold (Optional, +10 min)
```bash
# Build full image with SimpleFold
docker build -t amd-protein:simplefold -f Dockerfile .

# Run with SimpleFold
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  --shm-size=32g \
  -v $(pwd):/workspace \
  amd-protein:simplefold bash

# Inside container, run installation test
python test_simplefold_install.py

# Optional: Run inference test (downloads model, takes 5-10 min)
RUN_INFERENCE_TEST=true python test_simplefold_install.py
```

## Expected Results

✅ **Success Indicators:**
- ROCm detects MI300X GPU (192 GB)
- Docker builds without errors
- PyTorch detects GPU inside container
- All package imports work
- SimpleFold installation test passes (6/6 or 7/7)

❌ **Common Issues:**

**GPU not detected:**
```bash
# Check GPU devices exist
ls -l /dev/kfd /dev/dri/render*

# Check ROCm
rocm-smi

# Verify Docker has GPU access
docker run --rm --device=/dev/kfd --device=/dev/dri rocm/pytorch:latest rocm-smi
```

**Docker build fails:**
```bash
# Check disk space
df -h

# Check Docker daemon
systemctl status docker

# Try with more memory
docker build --memory=8g -f Dockerfile .
```

**Out of memory:**
```bash
# Use lightweight image
docker build -f Dockerfile.lightweight .

# Reduce shared memory
--shm-size=8g  # Instead of 32g
```

## Performance Benchmark (Optional)

Test inference speed on actual hardware:

```bash
# Inside container with SimpleFold
time simplefold \
  --simplefold_model simplefold_100M \
  --num_steps 200 \
  --tau 0.01 \
  --fasta_path test_data/test_short.fasta \
  --output_dir test_outputs \
  --backend torch

# Should complete in ~30-60 seconds
# Use this to calibrate time estimates for your hardware
```

## Production Deployment

Once tests pass:

```bash
# 1. Process sequences (already done locally)
# Just upload: sequences_by_length/

# 2. Start long-running job
docker-compose up -d amd-protein-predict

# 3. Monitor progress
docker-compose logs -f amd-protein-predict

# 4. Check GPU usage
watch -n 1 rocm-smi
```

## Minimal Test Script

Save this as `cloud_test.sh`:

```bash
#!/bin/bash
set -e

echo "=== AMD MI300X Docker Test ==="

# Test 1: GPU
echo "1. Testing GPU..."
rocm-smi --showproductname || { echo "❌ GPU test failed"; exit 1; }

# Test 2: Docker
echo "2. Testing Docker..."
docker --version || { echo "❌ Docker not found"; exit 1; }

# Test 3: Build
echo "3. Building Docker image..."
docker build -t test:latest -f Dockerfile.lightweight . || { echo "❌ Build failed"; exit 1; }

# Test 4: Run
echo "4. Testing container..."
docker run --rm --device=/dev/kfd --device=/dev/dri test:latest \
  python -c "import torch; assert torch.cuda.is_available(); print('✓ GPU accessible in container')" || \
  { echo "❌ Container test failed"; exit 1; }

echo "✅ All tests passed! Ready for production."
```

Run with:
```bash
chmod +x cloud_test.sh
./cloud_test.sh
```

**Total test time: 15-20 minutes**
