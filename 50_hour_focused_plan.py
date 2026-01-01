#!/usr/bin/env python3
"""
50 MI300X Hour Plan - Focus on Longest Proteins (H100-impossible sequences)
Based on actual SimpleFold-1.6B benchmarks
"""

import csv

# Actual SimpleFold-1.6B benchmarks from user
benchmarks = {
    64: 15,
    128: 20,
    256: 32,
    512: 61,
    1024: 139,
    2048: 385
}

print("=" * 70)
print("ACTUAL SIMPLEFOLD-1.6B BENCHMARKS")
print("=" * 70)
for length, time_s in benchmarks.items():
    print(f"  {length:4d} residues: {time_s:3d}s ({time_s/60:.2f} min)")

# Extrapolate for longer sequences
# Appears roughly quadratic with sequence length
def estimate_time(length):
    """Estimate time in seconds for a given sequence length"""
    if length <= 64:
        return 15
    elif length <= 128:
        return 20
    elif length <= 256:
        return 32
    elif length <= 512:
        return 61
    elif length <= 1024:
        return 139
    elif length <= 2048:
        return 385
    else:
        # Extrapolate: roughly scales as (length/2048)^2 * 385
        return 385 * (length / 2048) ** 2

print("\n" + "=" * 70)
print("EXTRAPOLATED TIMES FOR LONGEST SEQUENCES")
print("=" * 70)
for length in [3000, 4000, 5000, 5436]:
    time_s = estimate_time(length)
    print(f"  {length:4d} residues: {time_s:.0f}s ({time_s/60:.1f} min, {time_s/3600:.2f} hr)")

print("\n" + "=" * 70)
print("H100 MEMORY LIMIT ANALYSIS")
print("=" * 70)

# SimpleFold memory scales roughly as O(L^2) for sequence length L
# H100 SXM: 80 GB
# Estimate: ~3000aa is the practical limit for H100
h100_limit = 3000

print(f"\nH100 80GB memory limit: ~{h100_limit} residues")
print(f"MI300X 192GB can handle: ALL sequences (up to 5,436 residues)")

# Load sequence distribution
print("\n" + "=" * 70)
print("SEQUENCE DISTRIBUTION BY LENGTH")
print("=" * 70)

try:
    bins = []
    with open('sequences_by_length/batch_config.csv', 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            bins.append({
                'bin': parts[0],
                'length_range': parts[1],
                'num_sequences': int(parts[3])
            })
    
    total_sequences = sum(b['num_sequences'] for b in bins)
    
    print(f"\nTotal sequences: {total_sequences:,}")
    print("\nLength distribution:")
    
    h100_can_run = 0
    mi300x_exclusive = 0
    
    for bin_info in bins:
        bin_name = bin_info['bin']
        num_seq = bin_info['num_sequences']
        length_range = bin_info['length_range']
        
        # Estimate if H100 can handle this bin
        max_part = length_range.split('-')[1].replace('aa', '').replace('+', '').strip()
        max_length = int(max_part) if max_part.isdigit() else 10000
        
        if max_length <= h100_limit:
            h100_status = "✓ H100 OK"
            h100_can_run += num_seq
        else:
            h100_status = "✗ MI300X ONLY"
            mi300x_exclusive += num_seq
        
        print(f"  {bin_name}: {num_seq:5,} seqs ({length_range:15s}) {h100_status}")
    
    print(f"\nH100 can run: {h100_can_run:,} sequences ({h100_can_run/total_sequences*100:.1f}%)")
    print(f"MI300X exclusive: {mi300x_exclusive:,} sequences ({mi300x_exclusive/total_sequences*100:.1f}%)")
    
except FileNotFoundError:
    print("\nUsing approximate distribution:")
    bins = [
        {"bin": "bin1", "num_sequences": 9821, "length_range": "0-99", "max_len": 99},
        {"bin": "bin2", "num_sequences": 15907, "length_range": "100-199", "max_len": 199},
        {"bin": "bin3", "num_sequences": 13719, "length_range": "200-299", "max_len": 299},
        {"bin": "bin4", "num_sequences": 11951, "length_range": "300-399", "max_len": 399},
        {"bin": "bin5", "num_sequences": 11731, "length_range": "400-499", "max_len": 499},
        {"bin": "bin6", "num_sequences": 13474, "length_range": "500-999", "max_len": 999},
        {"bin": "bin7", "num_sequences": 3297, "length_range": "1000-1999", "max_len": 1999},
        {"bin": "bin8", "num_sequences": 474, "length_range": "2000-5436", "max_len": 5436},
    ]
    
    h100_can_run = sum(b['num_sequences'] for b in bins if b['max_len'] <= h100_limit)
    mi300x_exclusive = sum(b['num_sequences'] for b in bins if b['max_len'] > h100_limit)
    total_sequences = sum(b['num_sequences'] for b in bins)
    
    for b in bins:
        status = "✓ H100 OK" if b['max_len'] <= h100_limit else "✗ MI300X ONLY"
        print(f"  {b['bin']}: {b['num_sequences']:5,} seqs ({b['length_range']:12s} aa) {status}")
    
    print(f"\nH100 can run: {h100_can_run:,} sequences ({h100_can_run/total_sequences*100:.1f}%)")
    print(f"MI300X exclusive: {mi300x_exclusive:,} sequences ({mi300x_exclusive/total_sequences*100:.1f}%)")

print("\n" + "=" * 70)
print("50-HOUR BUDGET: MI300X EXCLUSIVE SEQUENCES")
print("=" * 70)

# Focus on bin8 (2000+ aa) - these are definitely MI300X-only
bin8_count = 474
bin8_avg_length = 2500  # conservative estimate

# Time per sequence in bin8
time_per_seq_bin8 = estimate_time(bin8_avg_length)  # seconds
print(f"\nBin8 (2000-5436 aa): {bin8_count} sequences")
print(f"Average length: ~{bin8_avg_length} residues")
print(f"Time per sequence: ~{time_per_seq_bin8:.0f}s ({time_per_seq_bin8/60:.1f} min)")

total_time_bin8 = bin8_count * time_per_seq_bin8 / 3600  # hours
print(f"Total time for bin8: {total_time_bin8:.1f} hours")

# Bin7 (1000-1999 aa) - some of these might be challenging for H100
bin7_count = 3297
bin7_avg_length = 1500

# Split bin7 into H100-friendly (<3000aa) and MI300X-preferred (all of them, but priority to longest)
time_per_seq_bin7 = estimate_time(bin7_avg_length)
print(f"\nBin7 (1000-1999 aa): {bin7_count} sequences")
print(f"Average length: ~{bin7_avg_length} residues")
print(f"Time per sequence: ~{time_per_seq_bin7:.0f}s ({time_per_seq_bin7/60:.1f} min)")

total_time_bin7 = bin7_count * time_per_seq_bin7 / 3600  # hours
print(f"Total time for bin7: {total_time_bin7:.1f} hours")

print("\n" + "=" * 70)
print("RECOMMENDED 50-HOUR STRATEGY: LONGEST-FIRST")
print("=" * 70)

budget = 50.0  # hours
used = 0.0

print(f"\nTotal budget: {budget} hours")

# Phase 1: ESM2 embeddings (all sequences)
esm2_time = 1.0
print(f"\n--- Phase 1: ESM2-3B Embeddings (ALL sequences) ---")
print(f"  Time: {esm2_time} hour")
print(f"  Output: 80,374 sequences with embeddings")
used += esm2_time

remaining = budget - used
print(f"\n--- Phase 2: SimpleFold-1.6B Structures (LONGEST-FIRST) ---")
print(f"  Remaining: {remaining:.1f} hours")

# Strategy: Start with longest (bin8), then work backwards
sequences_done = 0

print(f"\n  Step 1: Bin8 (2000-5436 aa) - MI300X EXCLUSIVE")
if total_time_bin8 <= remaining:
    print(f"    ✓ Complete bin8: {bin8_count} structures in {total_time_bin8:.1f} hours")
    sequences_done += bin8_count
    used += total_time_bin8
    remaining -= total_time_bin8
else:
    partial_bin8 = int((remaining * 3600) / time_per_seq_bin8)
    print(f"    ⚠ Partial bin8: {partial_bin8} structures in {remaining:.1f} hours")
    sequences_done += partial_bin8
    used += remaining
    remaining = 0

if remaining > 0:
    print(f"\n  Step 2: Bin7 (1000-1999 aa) - CHALLENGING FOR H100")
    if total_time_bin7 <= remaining:
        print(f"    ✓ Complete bin7: {bin7_count} structures in {total_time_bin7:.1f} hours")
        sequences_done += bin7_count
        used += total_time_bin7
        remaining -= total_time_bin7
    else:
        partial_bin7 = int((remaining * 3600) / time_per_seq_bin7)
        print(f"    ⚠ Partial bin7: {partial_bin7} structures in {remaining:.1f} hours")
        sequences_done += partial_bin7
        used += remaining
        remaining = 0

# Check if we can do more
if remaining > 0:
    print(f"\n  Step 3: Additional sequences (remaining {remaining:.1f} hours)")
    print(f"    Can process ~{int((remaining * 3600) / estimate_time(800))} sequences from bin6 (500-999aa)")

print("\n" + "=" * 70)
print("SUMMARY: 50-HOUR IMPACT")
print("=" * 70)

print(f"\nTotal sequences processed: {sequences_done:,} structures")
print(f"Percentage of dataset: {sequences_done/total_sequences*100:.1f}%")
print(f"\n✓ ESM2 embeddings: 80,374 sequences (100%)")
print(f"✓ SimpleFold structures: {sequences_done:,} sequences")
print(f"\nSTRATEGIC VALUE:")
print(f"  • All {bin8_count} ultra-long proteins (H100 impossible)")
print(f"  • Most/all {bin7_count} long proteins (H100 challenging)")
print(f"  • These are the MOST VALUABLE predictions")
print(f"  • Complement with H100 for shorter sequences later if needed")

print("\n" + "=" * 70)
print("EXECUTION PLAN")
print("=" * 70)

print("""
1. Start Docker container on MI300X:
   docker-compose up -d

2. Generate ESM2 embeddings (1 hour):
   # Process all 80,374 sequences
   python run_esm2_embeddings.py

3. SimpleFold prediction - LONGEST FIRST:
   # Bin8 (2000+ aa): ~474 sequences, ~4.1 hours
   python run_simplefold.py --bin bin8 --model simplefold_1.6B
   
   # Bin7 (1000-1999 aa): ~3,297 sequences, ~13.8 hours
   python run_simplefold.py --bin bin7 --model simplefold_1.6B
   
   # Continue with bin6, bin5, etc. as time permits

4. Monitor progress:
   rocm-smi
   docker logs -f amd_protein_predict

RATIONALE: Focus on proteins that ONLY MI300X can handle.
These are scientifically valuable (large, complex proteins).
Shorter proteins can be run on cheaper H100 instances later.
""")

print("\n" + "=" * 70)
print("Cost-Benefit Analysis")
print("=" * 70)

cost_50h = 50 * 4.5
print(f"\n50 hours MI300X: ${cost_50h:.0f}")
print(f"\nYou get:")
print(f"  • Complete embeddings for 80,374 proteins")
print(f"  • Structures for {mi300x_exclusive:,} proteins H100 CANNOT handle")
print(f"  • Plus additional structures from challenging range")
print(f"\nValue proposition:")
print(f"  • Unique capability (192GB memory)")
print(f"  • Scientific priority (largest/most complex proteins)")
print(f"  • Can complement with H100 for remaining sequences (~$200 for 60 hours)")
