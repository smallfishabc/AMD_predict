#!/bin/bash
# Minimal cloud test - runs in < 5 minutes

set -e

echo "=========================================="
echo "AMD MI300X Quick Test"
echo "=========================================="

echo ""
echo "✓ Testing GPU..."
rocm-smi --showproductname | head -5

echo ""
echo "✓ Building Docker (lightweight)..."
docker build -q -t test:latest -f Dockerfile.lightweight . && echo "Build successful"

echo ""
echo "✓ Testing GPU in container..."
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  test:latest \
  python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB')"

echo ""
echo "=========================================="
echo "✅ ALL TESTS PASSED - Ready for production"
echo "=========================================="
